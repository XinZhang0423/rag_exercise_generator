import sys
import yaml
import json
sys.path.append("/home/dalhxwlyjsuo/guest_zhangx/rag_exercise_generator")
from tqdm import tqdm
from dialogue_manager.manager import DialogueManager

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config



if __name__=="__main__":
    config = load_config("config.yaml")
    prompts = {
    "程序": "PROCEDURAL_PROMPT_zh",
    "比较": "COMPARATIVE_PROMPT_zh",
    "因果": "CAUSAL_PROMPT_zh",
    "条件": "CONDITIONAL_PROMPT_zh",
    "评估": "EVALUATIVE_PROMPT_zh",
    "预测": "PREDICTIVE_PROMPT_zh",
    "解释": "EXPLANATORY_PROMPT_zh"
    }
    dm=DialogueManager(config)
    index_to_docstore_id=dm.chunk_faiss_vectorstore.index_to_docstore_id
    total_chunk_nums=len(list(index_to_docstore_id.keys()))-140
    
    result=[]
    for idx in tqdm(range(70,total_chunk_nums)):
        chunk_dict={}
        chunk,vector=dm.load_random_chunk(idx=idx)
        is_intensive,reason=dm.is_knowledge_intensive(chunk)
        # 判断是否为知识密集
        if not is_intensive:
            print(is_intensive,reason)
            chunk_dict={'id':idx,'chunk':chunk,'is_intensive':is_intensive,'reason':reason,'label':'error','strategies':[],'ans':None}
            result.append(chunk_dict)
            continue 
        else:
            label=dm.label_chunk(chunk)
            if 'error' in label:
                continue
            questions = {}
            stragtegies=dm.determine_strategies(chunk)
            # 路由到不同的类型里面：
            for strategy, prompt in prompts.items():
                if strategy in stragtegies:
                    print(prompt)
                    questions[strategy] = dm.generate_question_simple(chunk, prompt)
            chunk_dict = {
                'id': idx,
                'chunk': chunk,
                'is_intensive': is_intensive,
                'reason': reason,
                'label': 'error',
                'strategies': stragtegies,
                'questions': questions,  # 独立保存的问题
                'ans': None
            }
            print(chunk_dict)
            result.append(chunk_dict)
            
        with open("/home/dalhxwlyjsuo/guest_zhangx/rag_exercise_generator/data/intro/intro_chunks.jsonl", 'a', encoding='utf-8') as f:
            f.write(json.dumps(chunk_dict, ensure_ascii=False) + "\n")
    # print(len(result))
    # with open("/home/dalhxwlyjsuo/guest_zhangx/rag_exercise_generator/data/intro/intro_chunks.jsonl",'w',encoding='utf-8') as f:
    #     for chunk in result:
    #         f.write(json.dumps(chunk,ensure_ascii=False)+"\n")
            

    
