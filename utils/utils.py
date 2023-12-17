
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

def get_random_city_country(name_surname_gender_mansion):
    """
    Returns random city and country for each individual
    
    params: df with [name, surname, mansion, mansion_id, description, salary]
    :return: two lists of random city and country for each individual
    """
    city_country = [random.sample(config.CITIES_COUNTRY, 1)[0] for _ in range(len(name_surname_gender_mansion))]
    return [city for city, _ in city_country], [country for _, country in city_country]

############################################################################################################
############################################################################################################

def get_remote_hybrid_office(name_surname_gender_mansion):
    """
    Returns random remote, hybrid or office for each individual
    
    params: df with [name, surname, mansion, mansion_id, description, salary]
    :return: a list of remote, hybrid or office for each individual
        # 0: remote
        # 1: hybrid
        # 2: office
    """
    return [random.randint(0, 2) for _ in range(len(name_surname_gender_mansion))]

############################################################################################################
############################################################################################################

def get_relocation(name_surname_gender_mansion):
    """
    Returns random willin to relocate for each individual
    
    params: df with [name, surname, mansion, mansion_id, description, salary]
    :return: a list of willin to relocate for each individual
        # 0: no
        # 1: yes
    """
    return [random.randint(0, 1) for _ in range(len(name_surname_gender_mansion))]

############################################################################################################
############################################################################################################

def get_name_companies(name_surname_gender_mansion):
    """
    Returns random name of companies for each individual
    
    params: df with [name, surname, mansion, mansion_id, description, salary]
    :return: a list of name of companies for each individual
    """
    return [random.sample(config.NAME_COMPANIES_SECTORS, 1)[0][0] for _ in range(len(name_surname_gender_mansion))]

############################################################################################################
############################################################################################################

def get_part_full_time(name_surname_gender_mansion):
    """
    Returns random part or full time for each individual
    
    params: df with [name, surname, mansion, mansion_id, description, salary]
    :return: a list of part or full time for each individual
        # 0: part time
        # 1: full time
    """
    return [random.randint(0, 1) for _ in range(len(name_surname_gender_mansion))]

############################################################################################################
############################################################################################################

def get_level_position(name_surname_gender_mansion):
    """
    Returns random junior or senior for each individual
    
    params: df with [name, surname, mansion, mansion_id, description, salary]
    :return: a list of junior or senior for each individual
        # 0: junior
        # 1: senior
    """
    return [random.randint(0, 1) for _ in range(len(name_surname_gender_mansion))]

############################################################################################################
############################################################################################################

def get_years_experience(name_surname_gender_mansion):
    """
    Returns random years of experience for each individual
    
    params: df with [name, surname, mansion, mansion_id, description, salary]
    :return: a list of years of experience for each individual

    """
    return [random.randint(0, 20) for _ in range(len(name_surname_gender_mansion))]

############################################################################################################
############################################################################################################

def get_education(name_surname_gender_mansion):
    """
    Returns random years of experience for each individual
    
    params: df with [name, surname, mansion, mansion_id, description, salary]
    :return: a list that indicates the education level for each individual
        # 0: high school
        # 1: bachelor
        # 2: master
        # 3: phd
        # 4: postdoc

    """
    return [random.randint(0, 4) for _ in range(len(name_surname_gender_mansion))]

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
    
    # add city and country
    city, country = get_random_city_country(name_surname_gender_mansion)
    name_surname_gender_mansion['city'] = city
    name_surname_gender_mansion['country'] = country
    
    del city, country
    
    # add remote, hybrid or office
    name_surname_gender_mansion['remote_hybrid_office'] = get_remote_hybrid_office(name_surname_gender_mansion)
    
    # add relocation
    name_surname_gender_mansion['relocation'] = get_relocation(name_surname_gender_mansion)
    
    # add name of companies
    name_surname_gender_mansion['current_company'] = get_name_companies(name_surname_gender_mansion)
    
    # add part or full time
    name_surname_gender_mansion['part_full_time'] = get_part_full_time(name_surname_gender_mansion)
    
    # add level position
    name_surname_gender_mansion['junior_senior'] = get_level_position(name_surname_gender_mansion)
    
    # add years of experience
    name_surname_gender_mansion['experience_years'] = get_years_experience(name_surname_gender_mansion)
    
    # add education
    name_surname_gender_mansion['level_education'] = get_education(name_surname_gender_mansion)
    
    return name_surname_gender_mansion