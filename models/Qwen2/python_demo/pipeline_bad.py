# encoding=utf8
import argparse
import time
from transformers import AutoTokenizer


class Qwen2():
    def __init__(self, args):
        # devid
        self.devices = [int(d) for d in args.devid.split(",")]

        # load tokenizer
        print("Load " + args.tokenizer_path + " ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_path, trust_remote_code=True
        )

        # warm up
        self.tokenizer.decode([0])

        # preprocess parameters, such as prompt & tokenizer
        self.system_prompt = "You are a helpful assistant."
        self.history = [{"role": "system", "content": self.system_prompt}]
        # self.EOS = self.tokenizer.eos_token_id
        self.EOS = -1
        self.enable_history = args.enable_history

        # load model
        self.load_model(args)

    def load_model(self, args):
        if len(self.devices) > 1:
            import chat_parallel
            self.model = chat_parallel.Qwen()
            self.model.init(
                self.devices,
                self.tokenizer.im_end_id,
                args.model_path
            )
        else:
            import chat
            self.model = chat.Qwen()
            self.model.init(self.devices, args.model_path)
            self.model.temperature = args.temperature
            self.model.top_p = args.top_p
            self.model.repeat_penalty = args.repeat_penalty
            self.model.repeat_last_n = args.repeat_last_n
            self.model.max_new_tokens = args.max_new_tokens
            self.model.generation_mode = args.generation_mode
            self.model.prompt_mode = args.prompt_mode
        self.SEQLEN = self.model.SEQLEN


    def clear(self):
        self.history = [{"role": "system", "content": self.system_prompt}]


    def update_history(self):
        if self.model.token_length >= self.SEQLEN:
            print("... (reach the maximal length)", flush=True, end="")
            self.history = [{"role": "system", "content": self.system_prompt}]
        else:
            self.history.append({"role": "assistant", "content": self.answer_cur})


    def encode_tokens(self, input_str):
        tokens = self.tokenizer(input_str).input_ids
        return tokens


    def chat(self):
        """
        Start a chat session.
        """
        tokens = self.encode_tokens("Human: \n你是一个知识点摘要小助手，请基于以下教师授课内容用简洁的语言概括这节课的关键知识点。注意：课程内容文本中可能会有一些错别字，请忽略带有错别字的知识点，基于内容清晰和不违背事实的内容分点输出知识点概要，总计不超过200字我希沃山东授先王帅今天的分享设定场景是教育新化论坛。\n二十余所中学的校长。\n教育信化专家一起来参会。\n那我们有十分钟的时间结合本次信息化论坛会议做主题的汇报，接下来呢。\n进入我的分享。\n各位领导大家好，我是希沃山东授权王帅，今天代表公司汇报的主题是 ai助理教育数字化展现英语实现。\n首先还是简单跟各位领导介绍一下希沃这边的情况。\n希沃利属于广州世业电子科技有限公司，是世源股份在教育领域的自主品牌。\n那截止二三年底内部数据的统计。\n我们在全国已经覆盖了八百万名的教师，二百八十人间的教室。\n再次再次感谢各位以来。\n一直以来的信赖与支持。\n随着教育性化的不断推进，教育数字化无疑是我国开辟教育发展新赛道和塑造教育发展新优势的重要突破口。\n而提升教研水平，加强教师队伍建设，更是教育数字化发展的重要方向。\n是提升教学质量的根本保障。\n那作为教育领域的头部品牌。\n在教育数字化发展的推进中，我们同样肩负着使命，一直推进思考与实践。\n为此，在去年十月份的全国教育装备展天津展，我们首次突破，以小机场的形式演绎了未来一节课。\n呈现秀 ai数据大模型在教学中的应用。\n其中，在课前备课是老师教研的重要环节。\n传统方式下，老师的备课需要大量的时间去收集信息和美化课件。\n而通过希沃白板与字源大模型的结合。\n老师输入章节或者是课程的名称，三分钟就可以生成一份优质的互动课件。\n同时呢，参照新课标。\n呃生成我们的大纲和教学建议。\n提供实用的备课资源包。\n那课中无需借助录播等一视频的采集工具，希沃第七代的交互智能平板自身升级了 ai视频模组。\n可无感知的采集课堂的语音和教学行为。\n在课后结合算力和大语言的模型。\n十分钟就可以快速的生成多维的课堂实录和数据，老师下课就可以得到本节课的课堂反馈。\n那下面就让我们通过一份真实的课堂报告，感受下 ai与小学教音相结合的应用。\n那首先呢，我们以三 d建模的形式将整堂课进行了还原，方便老师从不同的视角回溯课堂发生的细节。\n比如在这里我们可以看到蓝色轨迹呢，就是老师的活动和寻堂的轨迹。\n在这里我们可以看到老师主要的活动区域更多的是集中在了前端，对于后排学生的关注度相对较少。\n那如果老师看到这个就很直观。\n下次就能知道，对后面的同学要多加关注。\n那再比如，课堂中学生与老师进行了哪些互动。\n老师的提问还有一些反馈情况如何？\n我们在这里的话呢，直接可以点击学生的头像。\n就可以快速的跳转回到当时的视频切面，学习作者的想象方法。\n在任务二里，我们将完成单元。\n那同时呢，我们还提供了更多维度的教学数据和分析模型。\n比如说，教学时间的分析。\n像问答，对话分析。\n以及像弗兰德斯，互动分析教学。\n为教研提供实证辅助。\n那不仅如此， ai还会结合课程与新课当大纲的落实情况进行分析并给出建议。\n那大家现在所看到的课程总览。\n师生对话，课堂互动新课标落实的四个部分的内容。\n就都是由希沃的讲学大模型生成的多维的总结。\n那这里的总结呢。\n不是人工写的，也不是套用的模板，而是由 ai自动生成的。\n比如说。\n我们打开师生对话部分。\n那结合教学目标和知识性目标， ai会自动对师的提问进行分类，结合符合知识性目标的一些举例。\n并且呢，摘录可以避免的一些相关的问题，给出对应的原因，让老师对课堂师生对话的问题呢有所认知。\n那同时呢，针对于不合适的问题， a还会给出非常明确的优化建议。\n举个例子，比如说，在教学目标一中。\n对于字词的理解和学习。\n那优化前啊。\n老师的提问是你认识这个字吗？\n其中根据教学目标。\n学生需要认识和理解课文中的内容，所以老师不需要询问学生是否认识。\n优化后的题我们来看一下是看看这个字你之前遇到过它吗。\n那，如果有，记得在哪里见过他。\n这类问题呢，更有助于激发学生的兴趣。\n让学生参与到课堂的讨论和思考中。\n并且呢。\n在老师的引导下，更好的提高他们的理解和记忆。\n那这就是希沃在 ai教学教研方面的一些思考和和实践。\n以上的应用是如何实现的呢？\n首先，作为交互智能品牌领域的引领者，希望始终持续突破于创新。\n行业首倡的四个摄像头是专为教学 ai分析应用设计的。\n面对当前的造势，普通的一体机的摄像头很难拍得全拍得清，很难支撑 ai的分析，但是通过多个摄像头的组合。\n我们的整体化质可以达到八 k级。\n超宽的呃画幅影像也可以更好的把教室的画面完整的记录下来。\n那同时还内置了八阵内的麦克风。\n其专业的音频处理内置降噪算法。\n可以更好的有效过滤像风扇。\n空调等常见的教室杂音，把课堂中的语音也更清晰全面的录制下来。\n那摄像头，麦克风可以类比于我们人类的眼睛和耳朵。\n汽车采集的信息需要大码来进行处理。\n为此呢。\n希沃自研了教学大模型，参数规模在七十亿两级。\n那不同于我们所熟知的像 gppt等大模型的是，它是在教垂直领域的专业大模型。\n采用的是定向的训练方案。\n使用了教材教案等两千两百亿的训练资源素材。\n让模型更加的专业化。\n更针对教育行业。\n那为了更好的应用于课堂场景呢？\n我们还通过独有的技术方案，将整个模型知识的数长度拓展至了十六 k头看。\n这样就能将整堂课老师师生对话的全部的内容完整的记录下来，更好的输出给教学大模型进行分析。\n那其中可见教案，论文，教材。\n这些垂直的教育里面的训练素材和语料。\n相较于其他的大模型。\n在教学过程的还原分析和反馈上可以更好的保证严谨的输出，为教学教也提供实证支撑。\n但同时我们发现算力有成了最大的难题。\n解决训练算力的路径呢？\n呃。\n主要我们知道有两种，第一种的话就是网络传输，那速度太慢。\n第二种就是本地进行部署算力中心。\n直接将数据在本地进行训练，但是呢部署成本极高。\n那同时呢就是整个的费用和运维相对也会比较好，省耗力。\n所以针对以上的训练。\n呃算力路径其我提出了。\n云边协同的 ai大模型架构。\n内置的算锂模块，相当于三十个。\n在学校正常的网络条件下，这样我们就可以将分析一节课，产出报告的时间最快的压缩到。\n十分钟以内，教研的效率大幅度得到了提升。\n那同时呢。\n整体的成本相比于本地的部署。\n模式的话。\n节约了约百分之九十。\n此外呢，我们还支持多平台多终端灵活的访问大模型，最终为教育用户提供既好又省的本地化教育大模型应用于服务。\n目前呢。\n这套方案也是在全国多所学校都有应落地。\n并且呢也协同了学校打造了其特有的应用特色与成效。\n那比如说距离我们比较近的上海长宁区的石屯小学。\n学校与希沃围绕着课堂反馈合作了上海式的课题研究，完成了世纪课题的申报。\n那也在积极探索学校数字化转型的新方向。\n其中，学校通过向单人单课分析优化和多人多课比较分析两种主要的模式。\n在老师纵向个人成长过程中，以及呢老师之间的横向对与分析中，进行三十件两反思以及同课异构的数字化探索与提升。\n截止到目前的话呢，学校已经有七十余位教师借助该系统进行了教学模课。\n提升教学质量以及课堂的教研水平。\n那以上就是希我基于 ai助力教育数字化转型的应余实践的分享，感谢各位的耐心聆听。\n谢谢。\n\n\nAssistant: \n")
        # self.model.empty_kvcache()
        self.stream_answer(tokens)


        tokens = self.encode_tokens("Human: \n你是谁\n\nAssistant: \n")
        self.stream_answer(tokens)


    def stream_answer(self, tokens):
        """
        Stream the answer for the given tokens.
        """
        tok_num = 0
        self.answer_cur = ""
        self.answer_token = []

        # First token
        first_start = time.time()
        token = self.model.forward_first(tokens)
        first_end = time.time()
        # Following tokens
        while token != self.EOS and tok_num < self.model.max_new_tokens:
            word = self.tokenizer.decode(token, skip_special_tokens=True)
            self.answer_token += [token]
            print(word, flush=True, end="")
            tok_num += 1
            token = self.model.forward_next()
        self.answer_cur = self.tokenizer.decode(self.answer_token)
        
        # counting time
        next_end = time.time()
        first_duration = first_end - first_start
        next_duration = next_end - first_end
        tps = tok_num / next_duration

        print()
        print(f"FTL: {first_duration:.3f} s")
        print(f"TPS: {tps:.3f} token/s")


    ## For Web Demo
    def stream_predict(self, query):
        """
        Stream the prediction for the given query.
        """
        self.answer_cur = ""
        self.input_str = query
        tokens = self.encode_tokens()

        for answer_cur, history in self._generate_predictions(tokens):
            yield answer_cur, history


    def _generate_predictions(self, tokens):
        """
        Generate predictions for the given tokens.
        """
        # First token
        next_token = self.model.forward_first(tokens)
        output_tokens = [next_token]

        # Following tokens
        while True:
            next_token = self.model.forward_next()
            if next_token == self.EOS:
                break
            output_tokens += [next_token]
            self.answer_cur = self.tokenizer.decode(output_tokens)
            if self.model.token_length >= self.SEQLEN:
                self.update_history()
                yield self.answer_cur + "\n\n\nReached the maximum length; The history context has been cleared.", self.history
                break
            else:
                yield self.answer_cur, self.history

        self.update_history()


def main(args):
    model = Qwen2(args)
    model.chat()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str, required=False, default="Qwen2-1.5B-Instruct-for_qwen2_V1.3520seq.int4.1dev.bmodel", help='path to the bmodel file')
    parser.add_argument('-t', '--tokenizer_path', type=str, default="Qwen2-1.5B-Instruct", help='path to the tokenizer file')
    parser.add_argument('-d', '--devid', type=str, default='0', help='device ID to use')
    parser.add_argument('--temperature', type=float, default=0.5, help='temperature scaling factor for the likelihood distribution')
    parser.add_argument('--top_p', type=float, default=1.0, help='cumulative probability of token words to consider as a set of candidates')
    parser.add_argument('--repeat_penalty', type=float, default=1.0, help='penalty for repeated tokens')
    parser.add_argument('--repeat_last_n', type=int, default=32, help='repeat penalty for recent n tokens')
    parser.add_argument('--max_new_tokens', type=int, default=256, help='max new token length to generate')
    parser.add_argument('--generation_mode', type=str, choices=["greedy", "penalty_sample"], default="penalty_sample", help='mode for generating next token')
    parser.add_argument('--prompt_mode', type=str, choices=["prompted", "unprompted"], default="unprompted", help='use prompt format or original input')
    parser.add_argument('--enable_history', action='store_true', help="if set, enables storing of history memory.") 
    args = parser.parse_args()
    main(args)

