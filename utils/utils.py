
import utils.config as config
import pandas as pd
import random

from transformers import AutoTokenizer, AutoModel
import torch

from tqdm import tqdm

############################################################################################################
############################################################################################################

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
# device = 'cuda' if torch.cuda.is_available() else 'cpu

############################################################################################################
############################################################################################################

def get_category_df_mansion_txt_id_category():
    """
    Returns a dataframe with the mansions and their categories.

    """
    
    tokenizer = AutoTokenizer.from_pretrained(config.NAME_MODEL_TEXT_ENCODER)
    model = AutoModel.from_pretrained(config.NAME_MODEL_TEXT_ENCODER)
    model.eval()
    model.to(device)

    # load data and lowercase
    MANSIONS = [mansion.lower() for mansion in config.WORK_MANSIONS]

    # tokenize 
    MANSIONS = tokenizer(MANSIONS, return_tensors="pt", padding=True)
    CATEGORY_MANSIONS = tokenizer(config.CATEGORY_MANSIONS, return_tensors="pt", padding=True)

    MANSIONS_OUT = model(**MANSIONS.to(device))
    CATEGORY_MANSIONS_OUT = model(**CATEGORY_MANSIONS.to(device))

    # get the CLS of the last hidden state
    MANSIONS_OUT = MANSIONS_OUT.last_hidden_state[:, 0, :].detach()
    CATEGORY_MANSIONS_OUT = CATEGORY_MANSIONS_OUT.last_hidden_state[:, 0, :].detach()

    # compute cosine similarity and softmax
    similarity = (MANSIONS_OUT @ CATEGORY_MANSIONS_OUT.T).softmax(dim=-1)

    # for each mansion, find the category with the highest similarity
    similarity = similarity.cpu().numpy()
    similarity = pd.DataFrame(similarity)

    # assign category to each mansion based on highest similarity
    category = similarity.idxmax(axis=1)

    # get category text
    category_text = [config.id2category[i] for i in category]

    # dataframe with mansions and categories
    return pd.DataFrame({'mansion': config.WORK_MANSIONS, 'category': category_text, 'category_id': category})
    


############################################################################################################
############################################################################################################


def get_combinations_name_surname_gender_mansion(df_mansion_category, seed=42):
    """
    Returns a dataframe that for each individual contains
    ['name', 'surname', 'gender', 'mansion', 'category', 'category_id']
    
    """
    random.seed(42)
    return pd.DataFrame([(name, surname, gender,  \
        mansion, df_mansion_category[df_mansion_category['mansion'] == mansion].category.values[0], \
        df_mansion_category[df_mansion_category['mansion'] == mansion].category_id.values[0]) \
        for name, gender in tqdm(zip(config.UNIQUE_NAMES, config.GENDER_LABELS)) \
            for surname, mansion in zip(config.UNIQUE_SURNAMES, random.sample(config.WORK_MANSIONS, len(config.UNIQUE_SURNAMES)))], \
                columns=['name', 'surname', 'gender', 'mansion', 'category', 'category_id'])
    
    

############################################################################################################
############################################################################################################

def random_salary(name_surname_gender_mansion):
    """
    Return random salary
    """
    return torch.randint(700, 7000, (len(name_surname_gender_mansion),)).tolist()

############################################################################################################
############################################################################################################


def fill_slots_description(description, name, surname, mansion):
    """
    Fill slots in description with random values

    :param description: string
    :return: string
    
    """
    
    # extract slot inside { }
    slots_in_description = ['{' + item.split('}')[0] + '}' for item in description.split('{') if len(item.split('}')) > 1 ]

    # intersect with slot values
    slots_in_description = list(set(slots_in_description).intersection(set(config.SLOT_VALUES.keys())))

    # fill slots
    for slot in list(slots_in_description):
        description = ''.join([s + item for s, item in zip(description.split(slot), [random.sample(config.SLOT_VALUES[slot], 1)[0] for _ in range(len(description.split(slot))-1)])]) + description.split(slot)[-1]
    
    # fill name, surname, mansion
    description = description.replace('{NAME}', name)
    description = description.replace('{SURNAME}', surname)
    description = description.replace('{MANSION}', mansion)
        
    return description

############################################################################################################
############################################################################################################

def get_5_random_skills(name_surname_gender_mansion):
    """
    Returns 5 random skills for each individual
    
    params: df with [name, surname, mansion, mansion_id, description, salary]
    :return: list of 5 random skills for each individual
    """
    return [', '.join(random.sample(config.SKILLS, 5)) for _ in range(len(name_surname_gender_mansion))]

############################################################################################################
############################################################################################################

def fill_all_slots_salary(name_surname_gender_mansion):
    """
    Fill all slots in description with random attributes in the dictionary
    in the config file. Additionally, add salary to the df.
    
    params: df with [name, surname, mansion, mansion_id]
    :return: df with [name, surname, mansion, mansion_id, description, salary, skills]

    """
    
    # get all [name, surname, mansion, description]
    X_description = zip(name_surname_gender_mansion['name'], name_surname_gender_mansion['surname'], name_surname_gender_mansion['mansion'], [random.sample(config.DESCRIPTIONS, 1)[0] for _ in range(len(name_surname_gender_mansion['name']))])

    # fill slots
    X_description = [fill_slots_description(description, name, surname, mansion) for name, surname, mansion, description in tqdm(X_description)]   

    # add description to df name_surname_gender_mansion
    name_surname_gender_mansion['description'] = X_description
    
    # add salary
    name_surname_gender_mansion['salary'] = random_salary(name_surname_gender_mansion)
    
    # add skills
    name_surname_gender_mansion['skills'] = get_5_random_skills(name_surname_gender_mansion)
    
    return name_surname_gender_mansion