#include "NvInfer.h"
#include "NvInferRuntimeCommon.h"
#include "core/conversion/converters/converters.h"
#include "core/util/prelude.h"
#include "plugins/interpolate_plugin.h"
#include "torch/torch.h"

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

/*
 * Helper functions
 */
#if NV_TENSORRT_MAJOR < 7 || (NV_TENSORRT_MAJOR == 7 && NV_TENSORRT_MINOR < 1)
void create_plugin(ConversionCtx* ctx, const torch::jit::Node* n, nvinfer1::ITensor* in, const char* name,
                   std::vector<int64_t> in_shape, std::vector<int64_t> out_shape, std::vector<int64_t> out_size,
                   std::string mode) {
  LOG_WARNING("Interpolation layer will be run through ATen, not TensorRT. Performance may be lower than expected");

  auto creator = new plugins::InterpolatePluginCreator();
  auto plugin = creator->createPlugin(name, in_shape, out_shape, out_size, mode, false);

  auto resize_layer = ctx->net->addPluginV2(reinterpret_cast<nvinfer1::ITensor* const*>(&in), 1, *plugin);
  TRTORCH_CHECK(resize_layer, "Unable to create interpolation plugin from node" << *n);

  resize_layer->setName(util::node_info(n).c_str());

  auto layer_output = ctx->AssociateValueAndTensor(n->outputs()[0], resize_layer->getOutput(0));

  LOG_DEBUG("Output tensor shape: " << layer_output->getDimensions());
}
#endif

void resize_layer_size(ConversionCtx* ctx, const torch::jit::Node* n, nvinfer1::ITensor* in,
                       std::vector<int64_t> out_shape, std::vector<float> scales, nvinfer1::ResizeMode mode,
                       bool align_corners = false) {
  TRTORCH_CHECK((out_shape.size() > 0) ^ (scales.size() > 0), "only one of out_shape or scales should be defined");
  auto resize_layer = ctx->net->addResize(*in);
  TRTORCH_CHECK(resize_layer, "Unable to create interpolation (resizing) layer from node" << *n);

  if (out_shape.size() > 0) {
    auto th_dynamic_shape_mask = torch::zeros(out_shape.size(), torch::kInt32);
    auto th_static_shape_mask = torch::zeros(out_shape.size(), torch::kInt32);
    for (size_t idx = 0; idx < out_shape.size(); ++idx) {
      if (out_shape[idx] == -1) {
        th_dynamic_shape_mask[idx] = 1;
      } else {
        th_static_shape_mask[idx] = out_shape[idx];
      }
    }

    auto dynamic_shape_mask = tensor_to_const(ctx, th_dynamic_shape_mask);
    auto static_shape_mask = tensor_to_const(ctx, th_static_shape_mask);
    auto input_shape = ctx->net->addShape(*in)->getOutput(0);
    auto dynamic_shape =
        ctx->net->addElementWise(*input_shape, *dynamic_shape_mask, nvinfer1::ElementWiseOperation::kPROD)
            ->getOutput(0);
    auto target_output_shape =
        ctx->net->addElementWise(*dynamic_shape, *static_shape_mask, nvinfer1::ElementWiseOperation::kSUM)
            ->getOutput(0);
    resize_layer->setInput(1, *target_output_shape);
  } else {
    resize_layer->setScales(scales.data(), scales.size());
    if (align_corners) {
      LOG_WARNING("interpolate with align_corners and scale_factor works differently in TensorRT and PyTorch.");
    }
  }

  resize_layer->setResizeMode(mode);
  resize_layer->setName(util::node_info(n).c_str());

  // if interpolation mode is linear, align corners must have been set to true.
  // else, don't use align corners.
  if (mode == nvinfer1::ResizeMode::kLINEAR) {
#if NV_TENSORRT_MAJOR < 7 || (NV_TENSORRT_MAJOR == 7 && NV_TENSORRT_MINOR < 1)
    TRTORCH_CHECK(align_corners, "resize layer only support align_corner with TensorRT <= 7.0");
    resize_layer->setAlignCorners(true);
#else
    resize_layer->setAlignCorners(align_corners);
#endif
  }

  auto layer_output = ctx->AssociateValueAndTensor(n->outputs()[0], resize_layer->getOutput(0));

  LOG_DEBUG("Output tensor shape: " << layer_output->getDimensions());
}

/*
 * Interpolate Converter
 */

auto interpolate_registrations TRTORCH_UNUSED =
    RegisterNodeConversionPatterns()
        .pattern(
            {"aten::upsample_nearest1d(Tensor self, int[] output_size, float? scales=None) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto in = args[0].ITensor();
               auto in_shape = util::toVec(in->getDimensions());

               if (args[1].IValue()->isNone() && args[2].IValue()->isNone()) {
                 TRTORCH_THROW_ERROR("Unable to convert node: " << util::node_info(n)
                                                                << "\nOne of size or scale_factor should be defined");
               } else if (!args[2].IValue()->isNone()) {
                 // Case 1: user uses scales
                 float scale = args[2].IValue()->toDouble();
                 std::vector<float> padded_scales(in_shape.size(), 1);
                 padded_scales[padded_scales.size() - 1] = scale;
                 resize_layer_size(ctx, n, in, {}, padded_scales, nvinfer1::ResizeMode::kNEAREST);
               } else {
                 // Case 2: user uses output size
                 auto out_size = util::toVec(util::toDims(args[1].unwrapToIntList()));
                 TRTORCH_ASSERT(out_size.size() == 1,
                                "aten::upsample_nearest1d input Tensor and output size dimension mismatch");

                 auto out_shape = in_shape;
                 std::copy(out_size.begin(), out_size.end(), out_shape.begin() + (in_shape.size() - out_size.size()));
                 resize_layer_size(ctx, n, in, out_shape, {}, nvinfer1::ResizeMode::kNEAREST);
               }

               return true;
             }})
            .pattern(
            {"aten::upsample_nearest1d.vec(Tensor self, int[] output_size, float? scales=None) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                 auto in = args[0].ITensor();
                 auto in_shape = util::toVec(in->getDimensions());

                 if (args[1].IValue()->isNone() && args[2].IValue()->isNone()) {
                     TRTORCH_THROW_ERROR("Unable to convert node: " << util::node_info(n)
                                                                    << "\nOne of size or scale_factor should be defined");
                 } else if (!args[2].IValue()->isNone()) {
                     // Case 1: user uses scales
                     float scale = args[2].IValue()->toDouble();
                     std::vector<float> padded_scales(in_shape.size(), 1);
                     padded_scales[padded_scales.size() - 1] = scale;
                     resize_layer_size(ctx, n, in, {}, padded_scales, nvinfer1::ResizeMode::kNEAREST);
                 } else {
                     // Case 2: user uses output size
                     auto out_size = util::toVec(util::toDims(args[1].unwrapToIntList()));
                     TRTORCH_ASSERT(out_size.size() == 1,
                                    "aten::upsample_nearest1d input Tensor and output size dimension mismatch");

                     auto out_shape = in_shape;
                     std::copy(out_size.begin(), out_size.end(), out_shape.begin() + (in_shape.size() - out_size.size()));
                     resize_layer_size(ctx, n, in, out_shape, {}, nvinfer1::ResizeMode::kNEAREST);
                 }

                 return true;
             }})
        .pattern(
            {"aten::upsample_nearest2d.vec(Tensor self, int[2] output_size, float? scales_h=None, float? "
             "scales_w=None) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto in = args[0].ITensor();
               auto in_shape = util::toVec(in->getDimensions());

               if (args[1].IValue()->isNone() && (args[2].IValue()->isNone() || args[3].IValue()->isNone())) {
                 TRTORCH_THROW_ERROR("Unable to convert node: " << util::node_info(n)
                                                                << "\nOne of size or scale_factor should be defined");
               } else if (!args[2].IValue()->isNone() && !args[3].IValue()->isNone()) {
                 // Case 1: user uses scales
                 float scale_h = args[2].IValue()->toDouble();
                 float scale_w = args[3].IValue()->toDouble();
                 std::vector<float> padded_scales(in_shape.size(), 1);
                 padded_scales[padded_scales.size() - 2] = scale_h;
                 padded_scales[padded_scales.size() - 1] = scale_w;
                 resize_layer_size(ctx, n, in, {}, padded_scales, nvinfer1::ResizeMode::kNEAREST);
               } else {
                 // Case 2: user uses output size
                 auto out_size = util::toVec(util::toDims(args[1].unwrapToIntList()));
                 TRTORCH_ASSERT(out_size.size() == 2,
                                "aten::upsample_nearest2d input Tensor and output size dimension mismatch");

                 auto out_shape = in_shape;
                 std::copy(out_size.begin(), out_size.end(), out_shape.begin() + (in_shape.size() - out_size.size()));
                 resize_layer_size(ctx, n, in, out_shape, {}, nvinfer1::ResizeMode::kNEAREST);
               }

               return true;
             }})
        .pattern(
            {"aten::upsample_nearest2d(Tensor self, int[2] output_size, float? scales_h=None, float? scales_w=None) -> "
             "(Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto in = args[0].ITensor();
               auto in_shape = util::toVec(in->getDimensions());

               if (args[1].IValue()->isNone() && (args[2].IValue()->isNone() || args[3].IValue()->isNone())) {
                 TRTORCH_THROW_ERROR("Unable to convert node: " << util::node_info(n)
                                                                << "\nOne of size or scale_factor should be defined");
               } else if (!args[2].IValue()->isNone() && !args[3].IValue()->isNone()) {
                 // Case 1: user uses scales
                 float scale_h = args[2].IValue()->toDouble();
                 float scale_w = args[3].IValue()->toDouble();
                 std::vector<float> padded_scales(in_shape.size(), 1);
                 padded_scales[padded_scales.size() - 2] = scale_h;
                 padded_scales[padded_scales.size() - 1] = scale_w;
                 resize_layer_size(ctx, n, in, {}, padded_scales, nvinfer1::ResizeMode::kNEAREST);
               } else {
                 // Case 2: user uses output size
                 auto out_size = util::toVec(util::toDims(args[1].unwrapToIntList()));
                 TRTORCH_ASSERT(out_size.size() == 2,
                                "aten::upsample_nearest2d input Tensor and output size dimension mismatch");

                 auto out_shape = in_shape;
                 std::copy(out_size.begin(), out_size.end(), out_shape.begin() + (in_shape.size() - out_size.size()));
                 resize_layer_size(ctx, n, in, out_shape, {}, nvinfer1::ResizeMode::kNEAREST);
               }

               return true;
             }})
        .pattern(
            {"aten::upsample_nearest3d(Tensor self, int[3] output_size, float? scales_d=None, float? scales_h=None, "
             "float? scales_w=None) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto in = args[0].ITensor();
               auto in_shape = util::toVec(in->getDimensions());

               if (args[1].IValue()->isNone() &&
                   (args[2].IValue()->isNone() || args[3].IValue()->isNone() || args[4].IValue()->isNone())) {
                 TRTORCH_THROW_ERROR("Unable to convert node: " << util::node_info(n)
                                                                << "\nOne of size or scale_factor should be defined");
               } else if (!args[2].IValue()->isNone() && !args[3].IValue()->isNone() && !args[4].IValue()->isNone()) {
                 // Case 1: user uses scales
                 float scale_d = args[2].IValue()->toDouble();
                 float scale_h = args[3].IValue()->toDouble();
                 float scale_w = args[4].IValue()->toDouble();
                 std::vector<float> padded_scales(in_shape.size(), 1);
                 padded_scales[padded_scales.size() - 3] = scale_d;
                 padded_scales[padded_scales.size() - 2] = scale_h;
                 padded_scales[padded_scales.size() - 1] = scale_w;
                 resize_layer_size(ctx, n, in, {}, padded_scales, nvinfer1::ResizeMode::kNEAREST);
               } else {
                 // Case 2: user uses output size
                 auto out_size = util::toVec(util::toDims(args[1].unwrapToIntList()));
                 TRTORCH_ASSERT(out_size.size() == 3,
                                "aten::upsample_nearest3d input Tensor and output size dimension mismatch");

                 auto out_shape = in_shape;
                 std::copy(out_size.begin(), out_size.end(), out_shape.begin() + (in_shape.size() - out_size.size()));
                 resize_layer_size(ctx, n, in, out_shape, {}, nvinfer1::ResizeMode::kNEAREST);
               }

               return true;
             }})
        .pattern(
            {"aten::upsample_linear1d(Tensor self, int[] output_size, bool align_corners, float? scales) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto in = args[0].ITensor();
               auto in_shape = util::toVec(in->getDimensions());
               bool align_corners = args[2].unwrapToBool();

               if (args[1].IValue()->isNone() && args[3].IValue()->isNone()) {
                 TRTORCH_THROW_ERROR("Unable to convert node: " << util::node_info(n)
                                                                << "\nOne of size or scale_factor should be defined");
               } else if (!args[3].IValue()->isNone()) {
                 // Case 1: user uses scales
                 float scale = args[3].IValue()->toDouble();
                 std::vector<float> padded_scales(in_shape.size(), 1);
                 padded_scales[padded_scales.size() - 1] = scale;
#if NV_TENSORRT_MAJOR < 7 || (NV_TENSORRT_MAJOR == 7 && NV_TENSORRT_MINOR < 1)
                 if (!align_corners) {
                   TRTORCH_THROW_ERROR("Unable to convert node: "
                                       << util::node_info(n)
                                       << "\nupsample_linear1d only supports align_corner with TensorRT <= 7.0.");
                 } else {
                   resize_layer_size(ctx, n, in, {}, padded_scales, nvinfer1::ResizeMode::kLINEAR, true);
                 }
#else
                 resize_layer_size(ctx, n, in, {}, padded_scales, nvinfer1::ResizeMode::kLINEAR, align_corners);
#endif
               } else {
                 // Case 2: user uses output size
                 auto out_size = util::toVec(util::toDims(args[1].unwrapToIntList()));
                 TRTORCH_ASSERT(out_size.size() == 1,
                                "aten::upsample_linear1d input Tensor and output size dimension mismatch");

                 auto out_shape = in_shape;
                 std::copy(out_size.begin(), out_size.end(), out_shape.begin() + (in_shape.size() - out_size.size()));
#if NV_TENSORRT_MAJOR < 7 || (NV_TENSORRT_MAJOR == 7 && NV_TENSORRT_MINOR < 1)
                 if (!align_corners) {
                   // align_corners not supported in TensorRT, create plugin and
                   // run layer through PyTorch
                   create_plugin(ctx, n, in, "linear1d", in_shape, out_shape, out_size, std::string("linear"));
                 } else {
                   resize_layer_size(ctx, n, in, out_shape, {}, nvinfer1::ResizeMode::kLINEAR, true);
                 }
#else
                 resize_layer_size(ctx, n, in, out_shape, {}, nvinfer1::ResizeMode::kLINEAR, align_corners);
#endif
               }

               return true;
             }})
        .pattern(
            {"aten::upsample_bilinear2d(Tensor self, int[2] output_size, bool align_corners, float? scales_h=None, "
             "float? scales_w=None) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto in = args[0].ITensor();
               auto in_shape = util::toVec(in->getDimensions());
               bool align_corners = args[2].unwrapToBool();

               if (args[1].IValue()->isNone() && (args[3].IValue()->isNone() || args[4].IValue()->isNone())) {
                 TRTORCH_THROW_ERROR("Unable to convert node: " << util::node_info(n)
                                                                << "\nOne of size or scale_factor should be defined");
               } else if (!args[3].IValue()->isNone() && !args[4].IValue()->isNone()) {
                 // Case 1: user uses scales
                 float scale_h = args[3].IValue()->toDouble();
                 float scale_w = args[4].IValue()->toDouble();
                 std::vector<float> padded_scales(in_shape.size(), 1);
                 padded_scales[padded_scales.size() - 2] = scale_h;
                 padded_scales[padded_scales.size() - 1] = scale_w;
#if NV_TENSORRT_MAJOR < 7 || (NV_TENSORRT_MAJOR == 7 && NV_TENSORRT_MINOR < 1)
                 if (!align_corners) {
                   TRTORCH_THROW_ERROR("Unable to convert node: "
                                       << util::node_info(n)
                                       << "\nupsample_linear2d only supports align_corner with TensorRT <= 7.0.");
                 } else {
                   resize_layer_size(ctx, n, in, {}, padded_scales, nvinfer1::ResizeMode::kLINEAR, true);
                 }
#else
                 resize_layer_size(ctx, n, in, {}, padded_scales, nvinfer1::ResizeMode::kLINEAR, align_corners);
#endif
               } else {
                 // Case 2: user uses output size
                 auto out_size = util::toVec(util::toDims(args[1].unwrapToIntList()));

                 TRTORCH_ASSERT(out_size.size() == 2,
                                "aten::upsample_bilinear2d input Tensor and output size dimension mismatch");

                 auto out_shape = in_shape;
                 std::copy(out_size.begin(), out_size.end(), out_shape.begin() + (in_shape.size() - out_size.size()));

#if NV_TENSORRT_MAJOR < 7 || (NV_TENSORRT_MAJOR == 7 && NV_TENSORRT_MINOR < 1)
                 if (!align_corners) {
                   // align_corners not supported in TensorRT, create plugin and
                   // run layer through PyTorch
                   create_plugin(ctx, n, in, "bilinear2d", in_shape, out_shape, out_size, std::string("bilinear"));
                 } else {
                   resize_layer_size(ctx, n, in, out_shape, {}, nvinfer1::ResizeMode::kLINEAR, true);
                 }
#else
                 resize_layer_size(ctx, n, in, out_shape, {}, nvinfer1::ResizeMode::kLINEAR, align_corners);
#endif
               }

               return true;
             }})
        .pattern(
            {"aten::upsample_trilinear3d(Tensor self, int[3] output_size, bool align_corners, float? scales_d=None, "
             "float? scales_h=None, float? scales_w=None) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto in = args[0].ITensor();
               auto in_shape = util::toVec(in->getDimensions());
               bool align_corners = args[2].unwrapToBool();

               if (args[1].IValue()->isNone() &&
                   (args[3].IValue()->isNone() || args[4].IValue()->isNone() || args[5].IValue()->isNone())) {
                 TRTORCH_THROW_ERROR("Unable to convert node: " << util::node_info(n)
                                                                << "\nOne of size or scale_factor should be defined");
               } else if (!args[3].IValue()->isNone() && !args[4].IValue()->isNone() && !args[5].IValue()->isNone()) {
                 // Case 1: user uses scales
                 float scale_d = args[3].IValue()->toDouble();
                 float scale_h = args[4].IValue()->toDouble();
                 float scale_w = args[5].IValue()->toDouble();
                 std::vector<float> padded_scales(in_shape.size(), 1);
                 padded_scales[padded_scales.size() - 3] = scale_d;
                 padded_scales[padded_scales.size() - 2] = scale_h;
                 padded_scales[padded_scales.size() - 1] = scale_w;
#if NV_TENSORRT_MAJOR < 7 || (NV_TENSORRT_MAJOR == 7 && NV_TENSORRT_MINOR < 1)
                 if (!align_corners) {
                   TRTORCH_THROW_ERROR("Unable to convert node: "
                                       << util::node_info(n)
                                       << "\nupsample_linear3d only supports align_corner with TensorRT <= 7.0.");
                 } else {
                   resize_layer_size(ctx, n, in, {}, padded_scales, nvinfer1::ResizeMode::kLINEAR, true);
                 }
#else
                 resize_layer_size(ctx, n, in, {}, padded_scales, nvinfer1::ResizeMode::kLINEAR, align_corners);
#endif
               } else {
                 // Case 2: user uses output size
                 auto out_size = util::toVec(util::toDims(args[1].unwrapToIntList()));
                 TRTORCH_ASSERT(out_size.size() == 3,
                                "aten::upsample_trilinear3d input Tensor and output size dimension mismatch");

                 auto out_shape = in_shape;
                 std::copy(out_size.begin(), out_size.end(), out_shape.begin() + (in_shape.size() - out_size.size()));
#if NV_TENSORRT_MAJOR < 7 || (NV_TENSORRT_MAJOR == 7 && NV_TENSORRT_MINOR < 1)
                 if (!align_corners) {
                   // align_corners not supported in TensorRT, create plugin and
                   // run layer through PyTorch
                   create_plugin(ctx, n, in, "trilinear3d", in_shape, out_shape, out_size, std::string("trilinear"));
                 } else {
                   resize_layer_size(ctx, n, in, out_shape, {}, nvinfer1::ResizeMode::kLINEAR, true);
                 }
#else
                 resize_layer_size(ctx, n, in, out_shape, {}, nvinfer1::ResizeMode::kLINEAR, align_corners);
#endif
               }

               return true;
             }});

}  // namespace
}  // namespace impl
}  // namespace converters
}  // namespace conversion
}  // namespace core
}  // namespace trtorch