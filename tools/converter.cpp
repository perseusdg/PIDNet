#include "torch/script.h"
#include "torch_tensorrt/torch_tensorrt.h"
#include "torch_tensorrt/torch_tensorrt.h"
using namespace torch_tensorrt::torchscript;

int main(){
     torch::jit::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load("model.jit");
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }

    module.to(at::kCUDA);
    module.eval();
    auto in = torch::randn({1, 3, 640, 640}, {torch::kCUDA});
    auto input_sizes = std::vector<torch_tensorrt::Input>({in.sizes()});
    CompileSpec info(input_sizes);
    info.truncate_long_and_double = false;
    info.require_full_compilation = false;
    info.enabled_precisions.insert(torch::kFloat);
    auto trt_mod = compile(module, info);
    auto out = trt_mod.forward({in});
}