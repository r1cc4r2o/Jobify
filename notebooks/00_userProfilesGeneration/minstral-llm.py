import torch
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from langchain import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model_id = "mistralai/Mistral-7B-Instruct-v0.1"

model_4bit = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda",quantization_config=quantization_config, )
tokenizer = AutoTokenizer.from_pretrained(model_id)

pipeline = pipeline(
        "text-generation",
        model=model_4bit,
        tokenizer=tokenizer,
        use_cache=True,
        device_map="auto",
        max_length=600,
        do_sample=True,
        top_k=5,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
)

llm = HuggingFacePipeline(pipeline=pipeline)


template = """<s>[INST] You are a helpful, respectful and honest assistant. Answer exactly in few words from the context
Answer the question below from context below :
{context}
{question_manager}
{question} [/INST] </s>
"""

question_p = """The manager is asking this: Which is the education of John?"""
question = """The manager is asking what is in the context above, could you answare to his question given the context?"""


context_p = """
CANDIDATE 1 PROFILE
Name: John Doe


Biography:
Hello! I'm John Doe, a software developer passionate about technology and innovation. I enjoy coding, hiking, and exploring new places. Currently living in Cityville, I love meeting new people and making connections in the tech community. Feel free to reach out if you share similar interests!

Education:

Bachelor of Science in Computer Science, University of Techland (2012-2016)
Work Experience:

Software Developer, Tech Innovations Inc. (2016-present)


CANDIDATE 2 PROFILE
Name: Olivia Baker

Biography:
Hi there! I'm Olivia, a passionate baker with a love for creating delightful treats. From mouth-watering cupcakes to artisan bread, baking is not just my profession; it's my life! Join me on this sweet journey, and let's share the joy of homemade goodies together. 🍰🍪

Education:

Culinary Arts Degree, Baking Specialization, Sweetsville Culinary Institute (2004-2006)
Work Experience:

Head Baker, Sweet Delights Bakery (2006-present)
Baking Specialties:

"""


prompt = PromptTemplate(template=template, input_variables=["question","context"])
llm_chain = LLMChain(prompt=prompt, llm=llm)
response = llm_chain.run({"question_manager":question_p,"context":context_p, "question": question})


print(response)
