"""Fetch the conv layer shapes of each workload from its torchvision ONNX export on HuggingFace.

    conda run -n archx python zoo/agraph/designs/sc_cnn_dse/hf/fetch.py

    <name>.json   repo, revision, file, input shape, and one entry per Conv node:
                  R, C (ofmap height, width), M (out channels), K_h, K_w, Z (in channels), stride, group.

The ONNX files stay in the HuggingFace cache. Shapes come from ONNX shape inference with a
1x3x224x224 input. FC (Gemm/MatMul) layers are skipped.
"""

import json
from pathlib import Path

import onnx
from huggingface_hub import hf_hub_download

# name: repo, pinned revision, file
REPOS = {
    'vgg16': ['mobilint/VGG16', 'acd3b5ca1e1ab77a56ce477e2e5a373071ece392', 'vgg16_IMAGENET1K_V1.onnx'],
    'resnet18': ['mobilint/ResNet18', '5e1bc433e184e9a5ad59b9e05c980f909feb3aba', 'resnet18_IMAGENET1K_V1.onnx'],
    'resnet50': ['mobilint/ResNet50.tv1_in1k', '280f1ecf75ae93398b760f36c48b600d033eab73', 'resnet50_IMAGENET1K_V1.onnx'],
    'googlenet': ['mobilint/GoogLeNet', 'dd27a5ef680216443c71a00754865eb49594f755', 'googlenet_IMAGENET1K_V1.onnx'],
    'alexnet': ['mobilint/AlexNet', 'a8b204d2ce3395a3035abc22e359c8a867f81b5e', 'alexnet_IMAGENET1K_V1.onnx'],
}
INPUT = [1, 3, 224, 224]


def conv_layers(path: str) -> list:
    model = onnx.load(path)
    weights = {init.name: list(init.dims) for init in model.graph.initializer}
    graph_input = [i for i in model.graph.input if i.name not in weights][0]
    for dim, size in zip(graph_input.type.tensor_type.shape.dim, INPUT):
        dim.Clear()
        dim.dim_value = size
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
              for v in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output)}

    layers = []
    for node in model.graph.node:
        if node.op_type != 'Conv':
            continue
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        group = attrs.get('group', 1)
        assert group == 1, f'{node.name}: group={group}, grouped conv is not modeled'
        _, Z, _, _ = shapes[node.input[0]]
        _, M, R, C = shapes[node.output[0]]
        M_w, Z_w, K_h, K_w = weights[node.input[1]]
        assert (M_w, Z_w) == (M, Z), f'{node.name}: weight {weights[node.input[1]]} does not match in {Z} out {M}'
        stride = attrs.get('strides', [1, 1])
        assert stride[0] == stride[1], f'{node.name}: stride {stride}'
        layers.append({'name': node.name, 'R': R, 'C': C, 'M': M, 'K_h': K_h, 'K_w': K_w, 'Z': Z, 'stride': stride[0], 'group': group})
    return layers


here = Path(__file__).resolve().parent
for name, (repo, revision, file) in REPOS.items():
    layers = conv_layers(hf_hub_download(repo, file, revision=revision))
    (here / f'{name}.json').write_text(
        json.dumps({'repo': repo, 'revision': revision, 'file': file, 'input': INPUT, 'layers': layers}, indent=1) + '\n')
    macs = sum(l['R'] * l['C'] * l['M'] * l['K_h'] * l['K_w'] * l['Z'] for l in layers)
    print(f'{name}: {repo} @ {revision}, {len(layers)} conv layers, {macs:.4e} MACs')
