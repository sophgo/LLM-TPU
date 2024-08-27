# https://github.com/yuunnn-w/RWKV_Pytorch
import sys
import logging
import json
import onnx
from onnx_utils import get_onnx_tensor_proto_dtype, get_onnx_tensor_proto_shape, create_const_of_shape
from onnx_utils import del_onnx_initializers, insert_onnx_nodes, add_onnx_inits, del_onnx_nodes, shape_elem_num

SIZE_1MB = 1024 * 1024

COMPRESS_NODE_TYPES0 = ["Gather"]
COMPRESS_NODE_TYPES1 = ["Conv", "Gemm", "MatMul"]
COMPRESS_NODE_TYPES = COMPRESS_NODE_TYPES0 + COMPRESS_NODE_TYPES1
CONST_OF_SHAPE_VALUE = 0.01

DTYPE_BYTES = {
    onnx.TensorProto.FLOAT: 4,
    onnx.TensorProto.FLOAT16: 2,
    onnx.TensorProto.BFLOAT16: 2,
}


def compress_onnx_model(onnx_model, size_th_bytes=SIZE_1MB):
    graph = onnx_model.graph
    initializer = graph.initializer

    name_2_init_map = {}
    for init in initializer:
        name_2_init_map[init.name] = init

    new_nodes = []
    new_inits = []
    removed_inits = []

    for node in graph.node:
        if node.op_type not in COMPRESS_NODE_TYPES:
            continue
        if node.op_type in COMPRESS_NODE_TYPES0:
            init_name = node.input[0]
        elif node.op_type in COMPRESS_NODE_TYPES1:
            init_name = node.input[1]
        if init_name not in name_2_init_map:
            continue

        init = name_2_init_map[init_name]
        dtype = get_onnx_tensor_proto_dtype(init)
        shape = get_onnx_tensor_proto_shape(init)

        if dtype not in DTYPE_BYTES:
            continue

        dtype_bytes = DTYPE_BYTES[dtype]
        shape_elem = shape_elem_num(shape)
        if shape_elem * dtype_bytes <= size_th_bytes:
            continue

        global CONST_OF_SHAPE_VALUE
        node, shape_init = create_const_of_shape(
            shape=shape, dtype=dtype, value=CONST_OF_SHAPE_VALUE, output_name=init.name)
        CONST_OF_SHAPE_VALUE += 0.003

        removed_inits.append(init)
        new_nodes.append(node)
        new_inits.append(shape_init)

    replaced_tensor_names = [init.name for init in removed_inits]
    print("replaced_tensor_names:", replaced_tensor_names)
    del_onnx_initializers(graph, replaced_tensor_names)
    insert_onnx_nodes(graph, 0, new_nodes)
    add_onnx_inits(graph, new_inits)
    return onnx_model, removed_inits


def uncompress_onnx_model(onnx_model, removed_inits):
    onnx_model.graph.initializer.extend(removed_inits)
    replaced_tensor_names = [init.name for init in removed_inits]

    del_nodes = []
    for node in onnx_model.graph.node:
        if node.op_type != "ConstantOfShape":
            continue
        if node.output[0] in replaced_tensor_names:
            del_nodes.append(node)
    recover_replaced_tensors = [_node.output[0] for _node in del_nodes]
    print("recover_replaced_tensors:", recover_replaced_tensors)

    if len(replaced_tensor_names) != len(recover_replaced_tensors):
        logging.error("replaced_tensor_names len != recover_replaced_tensors len")
    if set(replaced_tensor_names) != set(recover_replaced_tensors):
        logging.error("replaced_tensor_names != recover_replaced_tensors")

    del_onnx_nodes(onnx_model.graph, del_nodes, del_node_init=False)
    return onnx_model
