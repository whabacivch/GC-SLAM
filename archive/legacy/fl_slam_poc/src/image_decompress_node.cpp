#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cv_bridge/cv_bridge.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

namespace
{
constexpr int kQueueDepth = 10;
constexpr double kDepthMmToM = 1000.0;

// Mirror the layout used by `compressed_depth_image_transport`:
// enum compressionFormat (int32) + float depthParam[2] => 12 bytes.
struct CompressedDepthConfigHeader
{
  std::int32_t format;
  float depth_param[2];
};
static_assert(sizeof(CompressedDepthConfigHeader) == 12, "Unexpected compressedDepth header size");

std::string to_lower(std::string value)
{
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

std::string trim(std::string value)
{
  auto not_space = [](unsigned char c) { return !std::isspace(c); };
  value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
  value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
  return value;
}

std::string parse_image_encoding(const std::string & format_field)
{
  const auto split_pos = format_field.find(';');
  if (split_pos == std::string::npos) {
    return trim(format_field);
  }
  return trim(format_field.substr(0, split_pos));
}

bool is_32f_encoding(const std::string & encoding)
{
  // Common: "32FC1" (from compressedDepth transport).
  // Be permissive in case of variants like "32FC1 " or "32FC1; ...".
  const auto enc = to_lower(trim(encoding));
  return enc.rfind("32f", 0) == 0;
}
}  // namespace

class ImageDecompressNode final : public rclcpp::Node
{
 public:
  ImageDecompressNode() : rclcpp::Node("image_decompress_cpp")
  {
    declare_parameter<std::string>("rgb_compressed_topic", "/camera/color/image_raw/compressed");
    declare_parameter<std::string>("depth_compressed_topic", "/camera/aligned_depth_to_color/image_raw/compressedDepth");
    declare_parameter<std::string>("rgb_output_topic", "/camera/image_raw");
    declare_parameter<std::string>("depth_output_topic", "/camera/depth/image_raw");
    declare_parameter<bool>("depth_scale_mm_to_m", true);
    declare_parameter<std::string>("qos_reliability", "reliable");

    rgb_in_ = get_parameter("rgb_compressed_topic").as_string();
    depth_in_ = get_parameter("depth_compressed_topic").as_string();
    rgb_out_ = get_parameter("rgb_output_topic").as_string();
    depth_out_ = get_parameter("depth_output_topic").as_string();
    depth_scale_ = get_parameter("depth_scale_mm_to_m").as_bool();
    qos_reliability_ = to_lower(get_parameter("qos_reliability").as_string());

    pub_rgb_ = create_publisher<sensor_msgs::msg::Image>(rgb_out_, kQueueDepth);

    auto qos_profiles = build_qos_profiles(qos_reliability_);
    for (const auto & qos : qos_profiles) {
      sub_rgb_.push_back(create_subscription<sensor_msgs::msg::CompressedImage>(
        rgb_in_, qos,
        std::bind(&ImageDecompressNode::on_rgb_compressed, this, std::placeholders::_1)));
    }

    const bool enable_depth = !depth_in_.empty();
    if (enable_depth) {
      pub_depth_ = create_publisher<sensor_msgs::msg::Image>(depth_out_, kQueueDepth);
      for (const auto & qos : qos_profiles) {
        sub_depth_.push_back(create_subscription<sensor_msgs::msg::CompressedImage>(
          depth_in_, qos,
          std::bind(&ImageDecompressNode::on_depth_compressed, this, std::placeholders::_1)));
      }
    } else {
      pub_depth_ = nullptr;
    }

    if (enable_depth) {
      RCLCPP_INFO(get_logger(),
                  "Image decompression node started:\n  RGB:   %s -> %s\n  Depth: %s -> %s\n  QoS reliability: %s",
                  rgb_in_.c_str(), rgb_out_.c_str(), depth_in_.c_str(), depth_out_.c_str(),
                  qos_reliability_.c_str());
    } else {
      RCLCPP_INFO(get_logger(),
                  "Image decompression node started (RGB only):\n  RGB: %s -> %s\n  Depth: disabled (empty topic)\n  QoS reliability: %s",
                  rgb_in_.c_str(), rgb_out_.c_str(), qos_reliability_.c_str());
    }
  }

 private:
  std::vector<rclcpp::QoS> build_qos_profiles(const std::string & reliability)
  {
    if (reliability == "both") {
      return {
        rclcpp::QoS(kQueueDepth).reliable(),
        rclcpp::QoS(kQueueDepth).best_effort(),
      };
    }
    if (reliability == "best_effort") {
      return {rclcpp::QoS(kQueueDepth).best_effort()};
    }
    if (reliability == "system_default") {
      return {rclcpp::QoS(kQueueDepth).reliability(rclcpp::ReliabilityPolicy::SystemDefault)};
    }
    return {rclcpp::QoS(kQueueDepth).reliable()};
  }

  bool is_duplicate(const std::string & key, const builtin_interfaces::msg::Time & stamp)
  {
    auto & last = last_stamps_[key];
    if (last.first == stamp.sec && last.second == stamp.nanosec) {
      return true;
    }
    last = {stamp.sec, stamp.nanosec};
    return false;
  }

  void on_rgb_compressed(const sensor_msgs::msg::CompressedImage::SharedPtr msg)
  {
    if (is_duplicate("rgb", msg->header.stamp)) {
      return;
    }

    try {
      auto bgr = decode_color(*msg);
      if (bgr.empty()) {
        record_error(rgb_errors_, "RGB decode returned empty image");
        return;
      }

      cv::Mat rgb;
      cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

      cv_bridge::CvImage out;
      out.header = msg->header;
      // The rest of the pipeline expects rgb8 (see `sensor_io.py`).
      out.encoding = "rgb8";
      out.image = rgb;
      pub_rgb_->publish(*out.toImageMsg());

      rgb_count_ += 1;
      if (rgb_count_ == 1) {
        RCLCPP_INFO(get_logger(), "First RGB image decompressed: %dx%d",
                    rgb.cols, rgb.rows);
      }
    } catch (const std::exception & exc) {
      record_error(rgb_errors_, std::string("RGB decompression failed: ") + exc.what());
    }
  }

  void on_depth_compressed(const sensor_msgs::msg::CompressedImage::SharedPtr msg)
  {
    if (pub_depth_ == nullptr) {
      return;
    }
    if (is_duplicate("depth", msg->header.stamp)) {
      return;
    }

    try {
      auto cv_depth = decode_depth(*msg);
      if (cv_depth.empty()) {
        record_error(depth_errors_, "Depth decode returned empty image");
        return;
      }

      cv::Mat depth_m;
      if (depth_scale_ && (cv_depth.type() == CV_16U || cv_depth.type() == CV_16S)) {
        cv_depth.convertTo(depth_m, CV_32F, 1.0 / kDepthMmToM);
      } else if (cv_depth.type() == CV_32F) {
        depth_m = cv_depth;
      } else {
        cv_depth.convertTo(depth_m, CV_32F);
      }

      cv_bridge::CvImage out;
      out.header = msg->header;
      out.encoding = "32FC1";
      out.image = depth_m;
      pub_depth_->publish(*out.toImageMsg());

      depth_count_ += 1;
      if (depth_count_ == 1) {
        RCLCPP_INFO(get_logger(), "First depth image decompressed: %dx%d",
                    depth_m.cols, depth_m.rows);
      }
    } catch (const std::exception & exc) {
      record_error(depth_errors_, std::string("Depth decompression failed: ") + exc.what());
    }
  }

  cv::Mat decode_color(const sensor_msgs::msg::CompressedImage & msg) const
  {
    const auto & data = msg.data;
    cv::Mat buffer(1, static_cast<int>(data.size()), CV_8UC1,
                   const_cast<unsigned char *>(data.data()));
    return cv::imdecode(buffer, cv::IMREAD_COLOR);
  }

  cv::Mat decode_depth(const sensor_msgs::msg::CompressedImage & msg) const
  {
    const auto & data = msg.data;

    auto decode_bytes_as_png = [&](const unsigned char * ptr, size_t len) -> cv::Mat {
      if (ptr == nullptr || len == 0) {
        return {};
      }
      cv::Mat buffer(1, static_cast<int>(len), CV_8UC1, const_cast<unsigned char *>(ptr));
      return cv::imdecode(buffer, cv::IMREAD_UNCHANGED);
    };

    const auto format_lower = to_lower(msg.format);
    const bool is_compressed_depth = (format_lower.find("compresseddepth") != std::string::npos);

    // Fast path: non-compressedDepth messages are regular JPEG/PNG payloads.
    if (!is_compressed_depth) {
      return decode_bytes_as_png(data.data(), data.size());
    }

    // compressedDepth transport payload is:
    // [ConfigHeader (12 bytes)] + [PNG (or other) compressed image data].
    // For 16UC1, PNG decodes directly to CV_16UC1.
    // For 32FC1, PNG decodes to inverse-depth CV_16UC1 which must be converted using depthParam.
    if (data.size() <= sizeof(CompressedDepthConfigHeader)) {
      // Some bags omit the header; attempt raw decode.
      return decode_bytes_as_png(data.data(), data.size());
    }

    CompressedDepthConfigHeader cfg{};
    std::memcpy(&cfg, data.data(), sizeof(cfg));
    const float depth_quant_a = cfg.depth_param[0];
    const float depth_quant_b = cfg.depth_param[1];

    const auto image_encoding = parse_image_encoding(msg.format);
    const bool want_32f = is_32f_encoding(image_encoding);

    const unsigned char * payload = data.data() + sizeof(cfg);
    const size_t payload_len = data.size() - sizeof(cfg);

    cv::Mat decoded = decode_bytes_as_png(payload, payload_len);
    if (decoded.empty()) {
      // Fallback: try without header (robustness to non-standard encoders).
      decoded = decode_bytes_as_png(data.data(), data.size());
    }
    if (decoded.empty()) {
      return {};
    }

    if (!want_32f) {
      // e.g. 16UC1 depth (commonly millimeters).
      return decoded;
    }

    // 32FC1 compressedDepth uses inverse depth quantization:
    // depth[m] = depthQuantA / (invDepth - depthQuantB), invDepth in uint16.
    if (decoded.type() != CV_16UC1) {
      // Unexpected, but attempt to continue by converting to 16U.
      cv::Mat tmp;
      decoded.convertTo(tmp, CV_16U);
      decoded = tmp;
    }

    cv::Mat depth(decoded.rows, decoded.cols, CV_32FC1);
    const auto nan = std::numeric_limits<float>::quiet_NaN();

    for (int r = 0; r < decoded.rows; ++r) {
      const std::uint16_t * inv_row = decoded.ptr<std::uint16_t>(r);
      float * out_row = depth.ptr<float>(r);
      for (int c = 0; c < decoded.cols; ++c) {
        const std::uint16_t inv = inv_row[c];
        if (inv == 0) {
          out_row[c] = nan;
        } else {
          out_row[c] = depth_quant_a / (static_cast<float>(inv) - depth_quant_b);
        }
      }
    }

    return depth;
  }

  void record_error(int & counter, const std::string & message)
  {
    counter += 1;
    if (counter <= 5) {
      RCLCPP_WARN(get_logger(), "%s", message.c_str());
    }
  }

  std::string rgb_in_;
  std::string depth_in_;
  std::string rgb_out_;
  std::string depth_out_;
  bool depth_scale_{true};
  std::string qos_reliability_;

  std::vector<rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr> sub_rgb_;
  std::vector<rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr> sub_depth_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_rgb_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_depth_;  // null if depth_compressed_topic empty

  std::unordered_map<std::string, std::pair<int32_t, uint32_t>> last_stamps_;

  int rgb_count_{0};
  int depth_count_{0};
  int rgb_errors_{0};
  int depth_errors_{0};
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ImageDecompressNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
