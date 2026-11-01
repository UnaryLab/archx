"""Convert SCALE-Sim's conv-net topologies into the conv layer tables sc_cnn_dse reads.

    conda run -n archx-agraph python zoo/agraph/designs/sc_cnn_dse/scale_sim_conv/convert.py

    <name>.json   repo, revision, file, and one entry per conv layer:
                  R, C (ofmap height, width), M (out channels), K_h, K_w, Z (in channels), stride, group,
                  H_in, W_in (SCALE-Sim's IFMAP height, width).

Topologies are read from the ../SCALE-Sim checkout at a pinned commit. SCALE-Sim has no VGG16, so its
layers come from ../hf/vgg16.json (torchvision's ONNX graph on HuggingFace). FC rows and the ResNet 1x1
shortcut convs are dropped, as in the paper's Fig. 9. SCALE-Sim lists the unpadded IFMAP, so R, C are the same-padding output size.
"""

import csv
import json
import re
import subprocess
from math import ceil
from pathlib import Path

SCALE_SIM = '../SCALE-Sim'
REPO = 'scalesim-project/SCALE-Sim'
REVISION = '9f98c4371055a54c75209c2e02b640b897550532'
# name: topology file, conv layers in the paper's Fig. 9
TOPOLOGIES = {
    'vgg16': ['../hf/vgg16.json', 13],
    'resnet18': ['topologies/conv_nets/Resnet18.csv', 17],
    'resnet50': ['topologies/conv_nets/Resnet50.csv', 49],
    'googlenet': ['topologies/conv_nets/Googlenet.csv', 57],
    'alexnet': ['topologies/conv_nets/alexnet.csv', 5],
}
SHORTCUT = re.compile(r'Conv\d_s|CB\ds')


def conv_layers(text: str) -> list:
    layers = []
    for row in list(csv.reader(text.splitlines()))[1:]:
        row = [field.strip() for field in row]
        if not row or not row[0]:
            continue
        name, H, W, K_h, K_w, Z, M, stride = row[0], *map(int, row[1:8])
        if name.upper().startswith('FC') or SHORTCUT.fullmatch(name):
            continue
        layers.append({'name': name, 'R': ceil(H / stride), 'C': ceil(W / stride), 'M': M, 'K_h': K_h, 'K_w': K_w,
                       'Z': Z, 'stride': stride, 'group': 1, 'H_in': H, 'W_in': W})
    return layers


here = Path(__file__).resolve().parent
for name, (file, count) in TOPOLOGIES.items():
    if name == 'vgg16':
        source = json.loads((here / file).read_text())
        repo, revision, file = source['repo'], source['revision'], source['file']
        # VGG16 convs are all stride 1, so the IFMAP size equals the ofmap size
        assert all(l['stride'] == 1 for l in source['layers'])
        layers = [dict(l, H_in=l['R'], W_in=l['C']) for l in source['layers']]
    else:
        repo, revision = REPO, REVISION
        text = subprocess.run(['git', '-C', SCALE_SIM, 'show', f'{REVISION}:{file}'], capture_output=True, text=True, check=True).stdout
        layers = conv_layers(text)
    assert len(layers) == count, f'{name}: {len(layers)} conv layers, expected {count}'
    (here / f'{name}.json').write_text(
        json.dumps({'repo': repo, 'revision': revision, 'file': file, 'layers': layers}, indent=1) + '\n')
    macs = sum(l['R'] * l['C'] * l['M'] * l['K_h'] * l['K_w'] * l['Z'] for l in layers)
    print(f'{name}: {repo} {file}, {len(layers)} conv layers, {macs:.4e} MACs')
