#include "core/conversion/converters/converters.h"
#include "core/util/prelude.h"
#include "torch/torch.h"

#include <string>
#include <vector>

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

auto group_norm_registrations TRTORCH_UNUSED = RegisterNodeConversionPatterns().pattern({
    R"SIG(aten::group_norm222222(Tensor input, int? num_groups, Tensor? weight,
                            Tensor? bias, float eps, bool cudnn_enabled) -> (Tensor))SIG",
                            [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
      auto input = args[0].ITensor(); // assumes non-static input Tensor
      auto orig_shape = input->getDimensions();
      auto shape = util::toVec(orig_shape);
      auto tensor_type = util::toATenDType(input->getType());
      LOG_DEBUG("input type is: "<< tensor_type);
      auto options = torch::TensorOptions().dtype(tensor_type);

      auto num_groups = args[1].unwrapToInt();
      LOG_DEBUG("num_groups: " << num_groups);


      at::Tensor gamma, beta;

      if (ctx->input_is_dynamic) {
        gamma = args[2].unwrapToTensor();
        beta = args[3].unwrapToTensor();
      } else {
        gamma = args[2].unwrapToTensor(at::full({shape}, 1, {options}));
        beta = args[3].unwrapToTensor(at::full({shape}, 0, {options}));
      }

      auto gamma_tensor = tensor_to_const(ctx, gamma);
      auto beta_tensor = tensor_to_const(ctx, beta);
      float eps = static_cast<float>(args[4].unwrapToDouble(1e-5f));
      LOG_DEBUG("eps: " << eps);

      LOG_DEBUG("cudnn disregarded");

      auto should_unpack = util::toVec(orig_shape).size() < 4;
      if (should_unpack) {
          // expand spatial dims from 1D to 2D
          auto new_shape = util::toDimsPad(util::toVec(orig_shape), 4);
          LOG_DEBUG(
                  "Input shape is less than 4D got: "
                  << orig_shape << ", inserting shuffle layer to reshape to 4D tensor shape: " << new_shape);
          auto in_shuffle = ctx->net->addShuffle(*input);
          in_shuffle->setReshapeDimensions(new_shape);
          in_shuffle->setName(std::string("[Reshape input to " + util::toStr(new_shape) + ']').c_str());
          input = in_shuffle->getOutput(0);
      }

      std::string pluginName = "GroupNormalizationPlugin";
      nvinfer1::PluginFieldCollection fc;
      std::vector<nvinfer1::PluginField> f;
      f.emplace_back(nvinfer1::PluginField("eps", &eps, nvinfer1::PluginFieldType::kFLOAT32, 1));
      f.emplace_back(nvinfer1::PluginField("num_groups", &num_groups, nvinfer1::PluginFieldType::kINT32, 1));

      fc.nbFields = f.size();
      fc.fields = f.data();
      nvinfer1::IPluginV2* pluginV2 = ctx->mPluginRegistry.at(pluginName)->createPlugin("gnorm", &fc);

      LOG_DEBUG("Number of output is: "<< pluginV2->getNbOutputs());

      std::vector<nvinfer1::ITensor*> inputs;

      inputs.push_back(input);
      inputs.push_back(gamma_tensor);
      inputs.push_back(beta_tensor);

      auto layer = ctx->net->addPluginV2(reinterpret_cast<nvinfer1::ITensor* const*>(&inputs[0]), int(inputs.size()), *pluginV2);
      layer->setName(util::node_info(n).c_str());
      auto layer_output = ctx->AssociateValueAndTensor(n->outputs()[0], layer->getOutput(0));
      LOG_DEBUG("Output tensor shape: " << layer_output->getDimensions());
      return true;
    }});

} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch
