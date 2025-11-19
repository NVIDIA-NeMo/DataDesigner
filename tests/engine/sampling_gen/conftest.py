# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
import random

from faker import Faker
import pandas as pd
import pytest

from data_designer.config.sampler_params import SamplerType
from data_designer.engine.sampling_gen.data_sources.base import DataSource
from data_designer.engine.sampling_gen.data_sources.sources import SamplerRegistry
from data_designer.engine.sampling_gen.people_gen import PeopleGenFaker, PeopleGenFromDataset
from data_designer.engine.sampling_gen.schema_builder import SchemaBuilder


def create_person_args():
    return dict(
        first_name="Jane",
        middle_name="Mary",
        last_name="Doe",
        age=30,
        sex="Female",
        ethnic_background="mexican",
        marital_status="married",
        education_level="bachelors",
        bachelors_field="education",
        occupation="teacher",
        zipcode="11201",
        unit="Apt A",
        street_number=123,
        street_name="Main St",
        city="Brooklyn",
        county="Kings County",
        region="NY",
        country="USA",
        locale="en_US",
        uuid="123e4567-e89b-12d3-a456-426614174000",
    )


person_with_personas_mock_records = [
    {
        "first_name": "Inez",
        "middle_name": "A",
        "last_name": "Newman",
        "sex": "Female",
        "age": 61,
        "zipcode": "53156",
        "street_number": 21,
        "street_name": "River Rd",
        "unit": "",
        "city": "Palmyra",
        "region": "WI",
        "county": "Jefferson County",
        "country": "USA",
        "ethnic_background": "white",
        "marital_status": "married_present",
        "education_level": "high_school",
        "bachelors_field": None,
        "occupation": "healthcare_support",
        "uuid": "51682c01-3e40-49d3-83c0-6d83c4d4765e",
        "locale": "en_US",
        "phone_number": "334-361-4009",
        "email_address": "inez.newman@gmail.com",
        "birth_date": "1963-11-24",
        "ssn": "395-79-8453",
        "detailed_occupation": "personal_care_aide",
        "skills_and_expertise": "Inez's practical nature (low openness) and people-focused career have honed her communication skills, patience, and adaptability. She is also skilled in basic first aid and administers medication under supervision, demonstrating her ability to learn and apply necessary knowledge despite her low conscientiousness.",
        "career_goals_and_ambitions": "Despite her spontaneity, Inez appreciates job stability. Her current role as a personal care aide, serving her community, satisfies her average extraversion, while her low conscientiousness and high neuroticism prevent her from pursuing higher education or management roles.",
        "hobbies_and_interests": "Inez enjoys simple pleasures like gardening and cooking traditional family recipes, reflecting her practicality (low openness) and appreciation for familiar customs. She also relishes quiet time alone, reading mysteries or listening to classical music, balancing her need for social interaction (average extraversion) with her critical, competitive nature (very low agreeableness).",
        "skills_and_expertise_list": "['effective communication', 'patience', 'adaptability', 'basic first aid', 'medication administration']",
        "hobbies_and_interests_list": "['gardening', 'cooking traditional recipes', 'reading mysteries', 'listening to classical music']",
        "concise_persona": "A steadfast, close-knit community caretaker who thrives on predictability, practicality, and simple pleasures, yet harbors a critical, competitive edge.",
        "detailed_persona": "Raised in a traditional, close-knit community with deep-rooted customs, this individual values familiarity and structure, reflected in their preference for practical tasks and simple hobbies. Their career in personal care, while demanding, provides the stability and social interaction they crave. Despite their high neuroticism, they find solace in routines and the comfort of their community. Their low agreeableness and critical nature make them discerning in their personal choices, but they remain dedicated to their community and the people they serve.",
        "professional_persona": "A dedicated personal care aide, excelling in communication and adaptability, yet resistant to organizational structures and management roles due to low conscientiousness.",
        "finance_persona": "Cautious with money, prioritizing practical needs over impulsive spending, and preferring stability over high-risk investments due to their preference for predictability.",
        "healthcare_persona": "Proactive in maintaining their health, appreciative of medical guidance, yet anxious and sensitive to health changes due to their high neuroticism.",
    },
    {
        "first_name": "Kathryn",
        "middle_name": "Nancy",
        "last_name": "Stires",
        "sex": "Female",
        "age": 57,
        "zipcode": "84003",
        "street_number": 290,
        "street_name": "Maple Ave",
        "unit": "",
        "city": "American Fork",
        "region": "UT",
        "county": "Utah County",
        "country": "USA",
        "ethnic_background": "white",
        "marital_status": "married_present",
        "education_level": "graduate",
        "bachelors_field": "arts_humanities",
        "occupation": "education_library",
        "uuid": "cf0404f2-c023-40e7-9e6a-4628295bd012",
        "locale": "en_US",
        "phone_number": "801-370-3480",
        "email_address": "kathrynnancystires@gmail.com",
        "birth_date": "1968-01-24",
        "ssn": "529-86-9526",
        "detailed_occupation": "teacher_or_instructor",
        "skills_and_expertise": "As a teacher with low conscientiousness, Kathryn is not one to overplan or micromanage. Instead, she excels at creating a warm, inclusive classroom environment that resonates with her students. Her deep understanding of her students' backgrounds and needs, coupled with her High neuroticism that makes her attentive to their emotional states, enables her to provide personalized support. Her graduate degree in arts and humanities, combined with her practical, low openness mindset, makes her proficient in teaching English and History, focusing on real-world applications.",
        "career_goals_and_ambitions": "Despite her High neuroticism that can make her anxious about change, Kathryn's strong agreeableness drives her towards improving her school's environment for her students, co-workers, and community. She actively participates in parent-teacher associations and local education boards to advocate for her students. Her ambition, however, is balanced by her low conscientiousness, making her more focused on immediate, tangible improvements rather than long-term, complex strategies. She dreams of eventually becoming a respected, long-serving principal in her community,arked by her dedication to creating a nurturing educational environment.",
        "hobbies_and_interests": "Kathryn enjoys simple, low-key hobbies that align with her average extraversion and very high agreeableness. She loves hosting potlucks and game nights for her neighbors and friends, fostering a sense of community. Her low openness makes her appreciate familiar activities, so she often embarks on nature walks or birdwatching in the nearby American Fork Canyon. She also enjoys reading, but prefers non-fiction books about local history or biographies of inspiring, community-focused figures.",
        "skills_and_expertise_list": "['expert in creating inclusive learning environments', 'skilled in teaching english and history', 'strong understanding of student emotional needs', 'exceptional community-building skills']",
        "hobbies_and_interests_list": "['hosting potlucks and game nights', 'nature walks and birdwatching', 'reading non-fiction about local history and inspiring figures', 'gardening']",
        "concise_persona": "A warm, community-focused educator who finds joy in simplicity, fostering growth through personal connections",
        "detailed_persona": "A product of her small-town upbringing, this individual embodies a practical, down-to-earth approach to life, preferring structure and predictability. Her strong agreeableness and high neuroticism make her deeply empathetic, always attuned to the emotions and needs of those around her. As a teacher, she excels in creating an inclusive classroom environment, her graduate degree in arts and humanities and her low openness guiding her to focus on real-world applications. She finds fulfillment in simple, low-key hobbies that nurture her community, such as hosting potlucks and nature walks. Her ambition is to improve her school's environment, driven by her desire to help others, yet balanced by her low conscientiousness, making her more focused on immediate, tangible improvements.",
        "professional_persona": "A dedicated educator who excels in creating nurturing learning environments, using her low openness and high neuroticism to personalize support for her students",
        "finance_persona": "Financially conservative, prioritizing stability and immediate needs over long-term investments, reflecting her practical and low openness nature",
        "healthcare_persona": "Proactive in preventive care, prioritizing mental well-being due to her high neuroticism, yet may struggle with consistent self-care routines due to her low conscientiousness",
    },
    {
        "first_name": "Michelle",
        "middle_name": "",
        "last_name": "Dupre",
        "sex": "Female",
        "age": 63,
        "zipcode": "60171",
        "street_number": 130,
        "street_name": "Rue St Julien",
        "unit": "",
        "city": "River Grove",
        "region": "IL",
        "county": "Cook County",
        "country": "USA",
        "ethnic_background": "white",
        "marital_status": "never_married",
        "education_level": "some_college",
        "bachelors_field": None,
        "occupation": "office_administrative_support",
        "uuid": "aeb59453-5e9a-4f36-a540-c31672d18fde",
        "locale": "en_US",
        "phone_number": "815-349-1859",
        "email_address": "michelledupre93@gmail.com",
        "birth_date": "1961-08-13",
        "ssn": "320-86-1835",
        "detailed_occupation": "secretary_or_administrative_assistant",
        "skills_and_expertise": "Michelle is a seasoned secretary with over four decades of experience in office management and administrative support. She is proficient in Microsoft Office Suite, with an exceptional ability to organize and manage databases using Access. Her strong typing speed and accuracy, along with her meticulous proofreading skills, ensure she consistently delivers high-quality work. Michelle's average conscientiousness level enables her to balance her tasks effectively, making her a reliable team player.",
        "career_goals_and_ambitions": "With her years of experience, Michelle seeks to become an office manager, where she can employ her organizational skills and experience to oversee day-to-day operations. Despite her average levels of extraversion and neuroticism, she is content with the stability of her current role and is not eager to take on high-risk challenges. Instead, she aims to gradually assume more responsibilities within her comfort zone, continuously improving her skills through relevant courses and workshops.",
        "hobbies_and_interests": "Michelle enjoys activities that allow her to interact with others while maintaining a sense of predictability and routine. She is an active member of her church choir, finding joy in harmonizing with others but also valuing the structure and dedication required for rehearsals. Her interest in genealogy reflects her low openness and average agreeableness, allowing her to explore her past in a practical and cooperative manner with family members. She also enjoys cooking traditional european dishes, often hosting small dinner parties for her friends and family.",
        "skills_and_expertise_list": "['proficient in microsoft office suite', 'strong typing speed and accuracy', 'database management using access', 'meticulous proofreading skills', 'strong organizational skills', 'reliable team player']",
        "hobbies_and_interests_list": "['church choir member', 'genealogy research', 'cooking traditional european dishes', 'hosting small dinner parties']",
        "concise_persona": "A steadfast organizer, she weaves tradition, routine, and Czech-Polish heritage into her roles and relationships",
        "detailed_persona": "Nurtured by her European roots in River Grove, she cherishes tradition and community, reflected in her meticulous planning of cultural events. With an average level of openness, she finds comfort in predictability, channeling this into her administrative career spanning four decades. As a seasoned secretary, her exceptional organizational skills and commitment to detail make her a reliable backbone of any team. Her average extraversion and neuroticism levels allow her to balance social interaction with self-care, ensuring she remains resilient in her roles. Michelle's career goals mirror her cautious yet dedicated nature, aiming to gradually assume more responsibilities within her comfort zone.",
        "professional_persona": "A dedicated office manager, her low openness drives her to maintain structured, predictable work environments, while her average conscientiousness ensures she consistently delivers high-quality work",
        "finance_persona": "Cautious with money, she values financial stability and practical investments, reflecting her low openness and average conscientiousness",
        "healthcare_persona": "Prioritizes preventive care and wellness routines, her approach to healthcare is structured and diligent, reflecting her average conscientiousness and low openness",
    },
]


def mock_load_person_generator_method(_, with_synthetic_personas: bool = False):
    """Mock the load_person_generator method for the WithPersonPGM class."""
    return mock_person_generator_loader(with_synthetic_personas)


def mock_person_generator_loader(locale: str = "en_US", with_synthetic_personas: bool = False):
    """Mock the person generator loader function that is passed to the DatasetGenerator in SamplingGen."""

    class MockPersonGenerator:
        def generate_samples(self, size: int = 1, **kwargs):
            person_fields = list(create_person_args().keys())
            if locale == "en_US":  # we only have en_US mocks for synthetic personas
                records = person_with_personas_mock_records
            if not with_synthetic_personas:
                records = [{k: v for k, v in record.items() if k in person_fields} for record in records]
            if "evidence" in kwargs:
                evidence = kwargs["evidence"]
                for record in records:
                    for k, v in evidence.items():
                        assert k in record, f"Invalid Person argument: {k}."
                        if isinstance(v, list):
                            record[k] = random.choice(v)
                        else:
                            record[k] = v
            return pd.DataFrame(random.choices(records, k=size))

    return MockPersonGenerator()


@pytest.fixture
def stub_default_samplers(stub_people_gen_resource) -> list[DataSource]:
    defaults = [
        (
            SamplerType.SCIPY,
            {"dist_name": "norm", "dist_params": {"loc": 0.0, "scale": 1.0}, "decimal_places": None},
        ),
        (SamplerType.BINOMIAL, {"n": 10, "p": 0.5}),
        (SamplerType.BERNOULLI, {"p": 0.5}),
        (
            SamplerType.BERNOULLI_MIXTURE,
            {"p": 0.5, "dist_name": "norm", "dist_params": {"loc": 0.0, "scale": 1.0}},
        ),
        (SamplerType.GAUSSIAN, {"mean": 0.0, "stddev": 1.0, "decimal_places": 5}),
        (SamplerType.POISSON, {"mean": 1.0}),
        (SamplerType.UNIFORM, {"low": 0.0, "high": 1.0, "decimal_places": 3}),
        (SamplerType.CATEGORY, {"values": ["a", "b", "c"], "weights": None}),
        (
            SamplerType.DATETIME,
            {
                "start": "2025-01-01T00:00:00",
                "end": "2025-12-01T00:00:00",
                "unit": "D",
            },
        ),
        (
            SamplerType.UUID,
            {"prefix": "ZZZ-", "short_form": True, "uppercase": True},
        ),
        (
            SamplerType.PERSON_FROM_FAKER,
            {
                "locale": "en_GB",
                "sex": None,
                "city": None,
                "age_range": [18, 100],
            },
        ),
        (
            SamplerType.PERSON,
            {
                "locale": "en_US",
                "sex": None,
                "city": None,
                "age_range": [18, 100],
                "with_synthetic_personas": False,
            },
        ),
    ]

    samplers = defaultdict(list)

    for sampler_type, params in defaults:
        samplers["params"].append(params)
        samplers["sampler_types"].append(sampler_type)
        samplers["sources"].append(
            SamplerRegistry.get_sampler(sampler_type)(params=params, people_gen_resource=stub_people_gen_resource)
        )

    return dict(samplers)


@pytest.fixture
def stub_people_gen_pgm():
    return PeopleGenFromDataset(
        engine=mock_person_generator_loader(with_synthetic_personas=False),
        locale="en_US",
    )


@pytest.fixture
def stub_people_gen_with_personas():
    return PeopleGenFromDataset(
        engine=mock_person_generator_loader(with_synthetic_personas=True),
        locale="en_US",
    )


@pytest.fixture
def stub_person_generator_loader():
    return mock_person_generator_loader


@pytest.fixture
def stub_people_gen_resource(stub_people_gen_pgm):
    return {
        "en_US": stub_people_gen_pgm,
        "en_GB_faker": PeopleGenFaker(Faker("en_GB"), "en_GB"),
    }


@pytest.fixture
def stub_schema(stub_default_samplers):
    builder = SchemaBuilder()
    for i, (sampler_type, params) in enumerate(
        zip(stub_default_samplers["sampler_types"], stub_default_samplers["params"], strict=False)
    ):
        builder.add_column(name=f"col_{i}", sampler_type=sampler_type, params=params)
    return builder.build()


@pytest.fixture
def stub_sampler_columns(stub_default_samplers):
    builder = SchemaBuilder()
    for i, (sampler_type, params) in enumerate(
        zip(stub_default_samplers["sampler_types"], stub_default_samplers["params"], strict=False)
    ):
        builder.add_column(name=f"col_{i}", sampler_type=sampler_type, params=params)
    return builder.to_sampler_columns()


@pytest.fixture
def stub_schema_builder():
    return SchemaBuilder()
