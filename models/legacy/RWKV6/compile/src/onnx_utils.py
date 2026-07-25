# https://github.com/yuunnn-w/RWKV_Pytorch
import json
import onnx
import numpy as np

NODE_INDICES = {}


def set_onnx_input_shape(onnx_model, shape_cfg):
    if not shape_cfg:
        return onnx_model
    if isinstance(shape_cfg, str):
        shape_cfg = json.loads(shape_cfg)

    graph = onnx_model.graph
    for _input in graph.input:
        if _input.name not in shape_cfg:
            continue
        tensor_shape_proto = _input.type.tensor_type.shape

        new_shape = shape_cfg[_input.name]
        # delete old shape
        elem_num = len(tensor_shape_proto.dim)
        for i in reversed(range(elem_num)):
            del tensor_shape_proto.dim[i]

        for i, d in enumerate(new_shape):
            dim = tensor_shape_proto.dim.add()
            if d is None:
                d = -1
            if d < -1:
                d = f"unk_{-d}"
            if isinstance(d, int):
                dim.dim_value = d
            elif isinstance(d, str):
                dim.dim_param = d
            else:
                raise ValueError(f"invalid shape: {new_shape}")
    return onnx_model


def del_onnx_nodes(graph, nodes, del_node_init=False):
    unused_init_names = []
    if del_node_init:
        init_names = [init.name for init in graph.initializer]
        for node in nodes:
            for in_name in node.input:
                if in_name in init_names:
                    unused_init_names.append(in_name)

    indices = []
    for idx, node in enumerate(graph.node):
        if node in nodes:
            indices.append(idx)
    indices = sorted(indices, reverse=True)
    for idx in indices:
        del graph.node[idx]

    if del_node_init:
        del_onnx_initializers(graph, unused_init_names)


def add_onnx_inits(graph, new_inits):
    del_init_names = [init.name for init in new_inits]
    del_onnx_initializers(graph, del_init_names)
    graph.initializer.extend(new_inits)


def del_onnx_initializers(graph, del_init_names):
    indices = []
    for idx, tensor_proto in enumerate(graph.initializer):
        if tensor_proto.name in del_init_names:
            indices.append(idx)

    indices = sorted(indices, reverse=True)
    for idx in indices:
        del graph.initializer[idx]


def insert_onnx_nodes(graph, idx, new_nodes):
    new_nodes = reversed(new_nodes)
    for node in new_nodes:
        graph.node.insert(idx, node)


def create_node_name(node_type):
    global NODE_INDICES
    if node_type not in NODE_INDICES:
        NODE_INDICES[node_type] = 0
    node_id = NODE_INDICES[node_type]
    NODE_INDICES[node_type] += 1

    name = f"{node_type}_{node_id}"
    return name


def create_const_of_shape(shape, dtype=onnx.TensorProto.FLOAT, value=0.0, output_name=None, node_name=None):
    if node_name is None:
        node_name = create_node_name("ConstantOfShape")
    if not output_name:
        output_name = node_name + "_output0"
    const_shape_name = node_name + "_shape"

    shape_dim = [len(shape)]
    shape_initializer = onnx.helper.make_tensor(
        name=const_shape_name, data_type=onnx.TensorProto.INT64, dims=shape_dim, vals=shape, raw=False)

    tensor_value_attr = onnx.helper.make_tensor("value", dtype, dims=[1], vals=[value])

    node = onnx.helper.make_node(op_type="ConstantOfShape",
                                 inputs=[const_shape_name],
                                 outputs=[output_name],
                                 value=tensor_value_attr)
    return node, shape_initializer


def get_onnx_tensor_proto_shape(onnx_tensor_proto):
    shape = [elem for elem in onnx_tensor_proto.dims]
    return shape


def get_onnx_tensor_proto_dtype(onnx_tensor_proto):
    return onnx_tensor_proto.data_type


def shape_elem_num(shape):
    elem_num = 1
    for elem in shape:
        elem_num *= elem
    return elem_num
