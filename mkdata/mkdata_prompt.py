
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

def response(content):
    
    return f"<response>\n{content}\n</response>\n"

def prompts(content, task = 'qa', language = 'zh'):
    return [{'role':'system','content':SYSTEMP_PROMPTS + "\n" + INSTRUCTIONS[task][language]['instructions'][0]},
            {'role':'user', 'content':  INSTRUCTIONS[task][language]['examples'][0]['user']},
             {'role':'assistant','content':INSTRUCTIONS[task][language]['examples'][0]['assistant']},
             ], [[{'role':'user','content': str(list(c))}] for c in content]


# def struct(title = "xxx", content = "xxx", next_action = "continue"):
#     return json.dumps({"title":title,
#                        "content":content,
#                        "next_action":next_action}) + "\n"

SYSTEMP_PROMPTS = "你是一个数据构造器，请根据我提供的原始信息和要求，严格按照要求生成我需要的数据。"

INSTRUCTIONS = {
    'qa': {
        # 'en':...,
        'zh':{
            'instructions': [
                f"我将会给你一个构造多跳问答数据的原始数据，即：由一个问题和多个形如 [头实体, 关系, 尾实体] 三元组组成的结构化数据，关系将头实体和尾实体联系在一起，通过对这些三元组的推导可以获得问题的答案，比如 ：  {information('[哈登, 专业特点, 欧洲步]')}。 \n 但是，可以发现information中的三元组并非自然语言。所以我希望你能够将information中的三元组组合成语句自然语言然后返回给我。字数要求：请用中文回复，要求生成的每句话不要太短(不低于20个字)，也不要太长(不超过100字)，尽量加一些修饰, 使生成的语句不至于太生硬,。生成的句子顺序要和三元组的排列顺序对应相同。每个三元组内的关系和实体在造句时顺序可以交换，也可以略微改变措辞，也可以多加一些无关信息，但不能改变和忽略核心语义。\n\n",
            ],
            # 'examples' : [
            #     examples(
            #         contexts(question('你知道休斯敦火箭队的知名人物有什么特点吗？') + \
            #                  information('[[休斯顿火箭队,知名人物,詹姆斯·哈登], [詹姆斯·哈登, 专业特点, 欧洲步]]') + \
            #                  answer('欧洲步')
            #                  ) + \
            #         response(
            #             information(
            #                 fact("休斯顿火箭的知名球星詹姆斯·哈登在近期对战洛杉矶湖人队的比赛中高效砍下30分。") + \
            #                 fact("哈登的技术特点是欧洲步，他在篮下使用欧洲步终结的命中率非常高。")
            #             )
            #         )
            #     )
            # ],

            'examples' : [{'user': '[休斯顿火箭队, 知名人物, 詹姆斯·哈登]',
                            'assistant': "休斯顿火箭的知名球星詹姆斯·哈登在近期对战洛杉矶湖人队的比赛中高效砍下30分。"
                            }
            ],

            'post_prompt': '下面我给一段原始数据，请按要求生成数据。'
        }

    }
}


PROMTS = {
    'qa' : [
        

    ],

    'counting_stars' :[

    ],

    # 'summary' : [

    # ]
}