import json

class Node(object):
    def __init__(self, node_name:str, op_name:str, stage:int=0):
        self.node_name = node_name
        self.op_name = op_name
        self.stage = stage

        self.input_subnet = []
        self.output_index = []
        self.output_tensorname = []
        self.output_stage = []

    def insert_input(self, node_name:str, out_index, output_tensorname: str, output_stage: int = 0):
        self.input_subnet.append(node_name)
        self.output_index.append(out_index)
        self.output_tensorname.append(output_tensorname)
        self.output_stage.append(output_stage)
        
    def get_json(self):
        node_data = {
            "net_name":self.node_name,
            "op_define":self.op_name,
            "stage":self.stage
        }
        input_data = []
        for index, input_name in enumerate(self.input_subnet):
            input_sub = {
                "subnet": input_name,
                "index":self.output_index[index],
                "tensorname":self.output_tensorname[index],
                "stage":self.output_stage[index]
            }
            input_data.append(input_sub)
        node_data["input"] = input_data
        return node_data


class Graph(object):
    def __init__(self):
        self.frist_nodes = []
        self.next_nodes = []
    
    def forward_first_insert(self, node: Node):
        self.frist_nodes.append(node)

    def forward_next_insert(self, node: Node):
        self.next_nodes.append(node)

    def save(self, file_name:str):
        file_obj = open(file_name,"w",encoding="utf-8")
        first_data = []
        next_data = []
        for node in self.frist_nodes:
            first_data.append(node.get_json())
        
        for node in self.next_nodes:
            next_data.append(node.get_json())
        data_dict = {
            "forward_first":first_data,
            "forward_next":next_data
        }
        json_data = json.dump(data_dict, file_obj, indent=2)
        file_obj.close()
