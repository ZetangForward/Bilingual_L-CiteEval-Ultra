from typing import Literal

TASK_TEMPLATE = "{instruction}\n\n{examples}\n\n{post_prompt}"
USER_TEMPLATE = '{source}'
DEFAULT_TEMPLATE = f'{TASK_TEMPLATE}\n\n{USER_TEMPLATE}'

def examples(content):
    return f"<example>\n{content}\n</example>\n\n"

def contexts(content, raw = False):
    return f"\n{content}\n" if raw else f"<context>\n{content}\n</context>\n" 

def question(content):    
    return f"<question>\n{content}\n</question>\n"

def information(content):
    return f"<information>\n{content}\n</information>\n"

def answer(content):
    return f"<answer>\n{content}\n</answer>\n"

def fact(content):
    return f"<fact>\n{content}\n</fact>\n"

def prompt(content):
    return f"<prompt>\n{content}\n</prompt>\n"



def response(content):
    
    return f"<response>\n{content}\n</response>\n"

def prompts(q, content, task = 'qa', language = 'zh', dtype :Literal['long', 'short', 'interference']= 'long'):
    return [{'role':'system','content':SYSTEMP_PROMPTS + "\n" + INSTRUCTIONS[task][language]['instructions'][dtype]},
            {'role':'user', 'content':  INSTRUCTIONS[task][language]['examples'][dtype]['user']},
             {'role':'assistant','content':INSTRUCTIONS[task][language]['examples'][dtype]['assistant']},
             ], [[{'role':'user','content': question(q) + str(list(c))}] for c in content]


def qa1_prompts(content, task = 'qa', language = 'zh', dtype = 'normal'):
    return [{'role':'system','content':SYSTEMP_PROMPTS + "\n" + MKQA1_INSTRUCTIONS[task][language]['instructions'][dtype]},
            {'role':'user', 'content':  MKQA1_INSTRUCTIONS[task][language]['examples'][dtype]['user']},
             {'role':'assistant','content':MKQA1_INSTRUCTIONS[task][language]['examples'][dtype]['assistant']},
               {'role':'user','content': str(content)}
             ]


def yesno_prompts(content, task = 'qa', language = 'zh', dtype = 'normal'):
    return [{'role':'system','content':SYSTEMP_PROMPTS + "\n" + MKYESNO_INSTRUCTIONS[task][language]['instructions'][dtype]},
            {'role':'user', 'content':  MKYESNO_INSTRUCTIONS[task][language]['examples'][dtype][0]['user']},
             {'role':'assistant','content':MKYESNO_INSTRUCTIONS[task][language]['examples'][dtype][0]['assistant']},
            {'role':'user', 'content':  MKYESNO_INSTRUCTIONS[task][language]['examples'][dtype][1]['user']},
             {'role':'assistant','content':MKYESNO_INSTRUCTIONS[task][language]['examples'][dtype][1]['assistant']},
               {'role':'user','content': str(content)}
             ]





# def struct(title = "xxx", content = "xxx", next_action = "continue"):
#     return json.dumps({"title":title,
#                        "content":content,
#                        "next_action":next_action}) + "\n"

SYSTEMP_PROMPTS = "你是一个数据构造器，请根据我提供的原始信息和要求，严格按照要求生成我需要的数据。"

INSTRUCTIONS = dict(
    qa = dict(
        # en =
        zh = dict(
            instructions = dict(
                long = f"我将会给你一个构造多跳问答数据的原始数据，即：由一个问题和多个形如 [头实体, 关系, 尾实体] 三元组组成的结构化数据，关系将头实体和尾实体联系在一起，通过对这些三元组的推导可以获得问题的答案，比如 ：  {question('休斯顿火箭队知名人物的专业特点是什么?') + contexts('[哈登, 专业特点, 欧洲步]')}。 \n 但是，可以发现information中的三元组并非自然语言。所以我希望你能够将information中的三元组组合成语句自然语言然后返回给我。逻辑要求：必须和问题相关。字数要求：请用中文回复，请尽可能长(至少300字)，尽量加一些修饰, 使生成的语句不至于太生硬。每个三元组内的关系和实体在造句时顺序可以交换，也可以略微改变措辞，也可以多加一些无关信息，但不能改变和忽略核心语义。\n\n",
                
                short = f"我将会给你一个构造多跳问答数据的原始数据，即：由一个问题和多个形如 [头实体, 关系, 尾实体] 三元组组成的结构化数据，关系将头实体和尾实体联系在一起，通过对这些三元组的推导可以获得问题的答案，比如 ：  {question('休斯顿火箭队知名人物的专业特点是什么?') + contexts('[哈登, 专业特点, 欧洲步]')}。 \n 但是，可以发现information中的三元组并非自然语言。所以我希望你能够将information中的三元组组合成语句自然语言然后返回给我。逻辑要求：只出现和我给你的三元组中相关的信息！我会依次给你多个三元组，但是你每次返回只能返回当前三元组相关的语句，不要使用上一次三元组中的内容！也不要输出任何与三元组无关的提示内容。字数要求：请用中文自然语言回复，请尽可能短（100字以内）。\n\n",

                interference = f"我将会给你一个构造多跳问答数据的原始数据，即：由一个问题和多个形如 [头实体, 关系, 尾实体] 三元组组成的结构化数据，关系将头实体和尾实体联系在一起，通过对这些三元组的推导可以获得问题的答案，比如 ：  {question('休斯顿火箭队知名人物的专业特点是什么?') + contexts('[哈登, 专业特点, 欧洲步]')}。 \n 但是，我需要你生成的是干扰数据，即和我给你的三元组领域类似但实际上完全不同的自然语句。逻辑要求：必须和问题的领域相近但对问题无关且对解答没有帮助,不许出现和三元组中的任何一个实体！。字数要求：请用中文回复，每句话请尽可能长，多加一些与问题领域相近但无关的修饰。格式要求：请生成20句与问题和三元组领域相似但无关的数据，每条数据用' ||| '隔开\n最后提醒，你生成的数据中应当不出现任何一个三元组的实体！\n\n",
            ),

            examples = dict(
                long = dict(
                    user = f"{question('休斯顿火箭队知名人物的专业特点是什么？')}[休斯顿火箭队, 知名人物, 哈登]",
                    assistant = "休斯顿火箭的知名球星哈登在近期对战洛杉矶湖人队的比赛中获得2篮板7助攻3抢断的数据，高效砍下30分,湖人队的外线防守未能有效限制其得分。"
                ),

                short = dict(
                    user = f"{question('休斯顿火箭队知名人物的专业特点是什么？')}[休斯顿火箭队, 知名人物, 哈登]",
                    assistant = "休斯顿火箭的知名人物是哈登。"
                ),

                
                interference = dict(
                    user = f"{question('休斯顿火箭队知名人物的专业特点是什么？')}[休斯顿火箭队, 知名人物, 哈登]",
                    assistant = "库里在对战湖人队时砍下30分，其中在对位詹姆斯时使用了后撤步三分压哨命中，这一球成为本场最佳球。 ||| 库里将于12月7号客场对战湖人队  ||| ... ||| 库里在客场对位篮网知名球星欧文时高效砍下33分，这是库里本赛季第一次与欧文交手"
                )
            ),

            post_prompt = '下面我给一段原始数据，请按要求生成数据。'
        )

    )
)


MKQA1_INSTRUCTIONS = dict(
    qa = dict(
        # en =
        zh = dict(
            instructions = dict(
                normal = f"我将会给你一个[实体1, 关系x, 实体2]的三元组，你现在需要根据这个三元组构造问题与答案。问题应当包括 实体1 和 关系x， 答案为实体2。请按照以下要求进行回复。格式要求：输出格式应为 {question('构造的问题')}。逻辑要求：问题应当只包含实体1和关系,不包含实体2，并且应当表示'和实体1有 关系x 关系的 实体是什么'。语言要求：请将问题构造成自然语言的。"
            ),

            examples = dict(
                normal = dict(
                    user = f"[哈登, 专业特点, 后撤步]",
                    assistant = f"{question('哈登的专业特点是什么？')}"
                )

            ),

            post_prompt = '下面我给一段原始数据，请按要求生成数据。'
        )

    )
)



MKYESNO_INSTRUCTIONS = dict(
    qa = dict(
        # en =
        zh = dict(
            instructions = dict(
                normal = f"我将会给你一个问题和相应的答案，你现在需要根据我给你的信息，将问题转换成 “是否” 类型的问题，我还会告诉你这些问题应当转为答案为“是”还是答案为“否”的问题（我会放在{prompt('')}之间），请根据要求转换，如果我提供：{prompt('是')}，那么你构造的问题的答案必须是我给你提供的答案，如果我提供：{prompt('否')}，那么你必须要改变给你的样例答案，即你生成的“是否类型”的问句必须不包含我给你的样例答案,请用仍然用肯定疑问，只是改变答案。请按照以下要求进行回复。格式要求：输出格式应为 {question('构造的问题')}。语言要求：请将问题构造成自然语言的疑问句。"
            ),

            examples = dict(
                normal = [
                    dict(
                    user = f"{question('哈登的专业特点是什么？') + answer('后撤步') + prompt('是')}",
                    assistant = f"哈登的专业特点是后撤步吗？"
                    ),
                    dict(
                        user = f"{question('请问八方横野的上司是谁？') + answer('贾命公') + prompt('否')}",
                        assistant = f"请问八方横野的上司是贾平凹吗？"
                    )
                    
                ]

            ),

            post_prompt = '下面我给一段原始数据，请按要求生成数据。'
        )

    )
)


PROMTS = {
    'qa' : [
        

    ],

    'counting_stars' :[

    ],

    # 'summary' : [

    # ]
}