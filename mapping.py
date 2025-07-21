import pandas as pd
from data_load import data_ingestion
df=data_ingestion()
def apply_custom_mappings(df):
    # Define all mappings
    country_mapping = {
        "India": "Tier 1", "United States": "Tier 1", "United Kingdom": "Tier 1", "Canada": "Tier 1",
        "Australia": "Tier 1", "Singapore": "Tier 1", "China": "Tier1",
        "United Arab Emirates": "Tier 2", "Saudi Arabia": "Tier 2", "Germany": "Tier 2",
        "France": "Tier 2", "Sweden": "Tier 2", "Hong Kong": "Tier 2",
        "Qatar": "Tier 3", "Oman": "Tier 3", "Bahrain": "Tier 3", "Kuwait": "Tier 3",
        "South Africa": "Tier 3", "Nigeria": "Tier 3",
        "unknown": "Unknown"
    }

    city_mapping = {
        "Mumbai": "Metro", "Thane & Outskirts": "Metro", "Other Metro Cities": "Metro",
        "Other Cities": "Non-Metro", "Tier II Cities": "Non-Metro", "Other Cities of Maharashtra": "Non-Metro",
        "Select": "Unknown"
    }

    lead_source_mapping = {
        "Google": "Search Engine", "google": "Search Engine", "bing": "Search Engine", "Organic Search": "Search Engine",
        "Direct Traffic": "Direct", "Welingak Website": "Website",
        "Reference": "Referral", "Referral Sites": "Referral",
        "Facebook": "Social Media", "youtubechannel": "Social Media", "Social Media": "Social Media",
        "Olark Chat": "Live Chat", "Live Chat": "Live Chat",
        "Click2call": "Paid Campaigns", "Pay per Click Ads": "Paid Campaigns",
        "Press_Release": "Media", "WeLearn": "Blog", "welearnblog_Home": "Blog", "blog": "Blog",
        "testone": "Other"
    }

    lead_profile_mapping = {
        "Student of SomeSchool": "Student", "Lateral student": "Student", "Dual Specialization Student": "Student",
        "Fresh Graduate": "Graduate", "Working Professional": "Professional", "Businessman": "Business",
        "Housewife": "Other", "Other Leads": "Other", "Select": "Unknown", "Not Specified": "Unknown"
    }

    last_notable_activity_mapping = {
        "Email Opened": "Engaged via Email", "Email Link Clicked": "Engaged via Email", "View in browser link Clicked": "Engaged via Email",
        "Resubscribed to emails": "Re-engaged", "Email Received": "Re-engaged",
        "Email Marked Spam": "Disengaged", "Unsubscribed": "Disengaged", "Email Bounced": "Disengaged",
        "Page Visited on Website": "Website Browsing", "Form Submitted on Website": "Website Browsing",
        "Olark Chat Conversation": "Active Interaction", "Had a Phone Conversation": "Active Interaction", "Approached upfront": "Active Interaction",
        "SMS Sent": "Contact Status", "Unreachable": "Contact Status",
        "Modified": "Profile Updated"
    }

    tags_mapping = {
        "Will revert after reading the email": "Low Intent", "Still Thinking": "Low Intent",
        "In confusion whether part time or DLP": "Undecided", "Interested in other courses": "Course Mismatch",
        "Ringing": "Trying to Contact", "Busy": "Trying to Contact", "Switched off": "Trying to Contact", "No Response": "Trying to Contact",
        "Wrong Number Given": "Invalid Contact", "Not Interested in Course": "Not Interested",
        "Lost to EINS": "Lost Lead", "Lost to Others": "Lost Lead",
        "Already a student": "Existing Student", "Closed by Horizzon": "Closed",
        "Not Specified": "Unknown", "Select": "Unknown"
    }

    specialization_mapping = {
        "Business Administration": "Business", "International Business": "Business", "Services Excellence": "Business",
        "Finance Management": "Finance", "Banking, Investment And Insurance": "Finance",
        "Human Resource Management": "HR", "Marketing Management": "Marketing", "Media and Advertising": "Marketing", "Digital Marketing": "Marketing",
        "Operations Management": "Operations", "Supply Chain Management": "Operations",
        "IT Projects Management": "IT", "E-Business": "IT", "E-Commerce": "IT",
        "Retail Management": "Retail", "Health Care Management": "Health",
        "Hospitality Management": "Hospitality", "Travel and Tourism": "Hospitality",
        "Rural and Agribusiness": "Agribusiness",
        "Select": "Unknown", "Not Specified": "Unknown"
    }

    lead_quality_mapping = {
        "High in Relevance": "High", "Best": "High",
        "Might be": "Medium", "Not Sure": "Unknown",
        "Low in Relevance": "Low", "Worst": "Low",
        "Select": "Unknown", "Not Specified": "Unknown"
    }

    occupation_mapping = {
        "Unemployed": "Unemployed", "Student": "Student", "Working Professional": "Professional",
        "Businessman": "Business", "Housewife": "Other", "Other": "Other",
        "Not Specified": "Unknown", "Select": "Unknown" 
    }

    hear_about_mapping = {
        "Word Of Mouth": "Referral", "Online Search": "Search Engine",
        "Advertisements": "Ads", "Digital Advertisement": "Ads",
        "Email Campaign": "Email", "Social Media": "Social Media",
        "Friend/Colleague": "Referral", "Others": "Other",
        "Unknown": "Unknown", "Not Mentioned": "Unknown", "Select": "Unknown"
    }

    # Apply all mappings
    df["Country"] = df["Country"].map(country_mapping)
    df["City"] = df["City"].map(city_mapping)
    df["Lead Source"] = df["Lead Source"].map(lead_source_mapping)
    df["Lead Profile"] = df["Lead Profile"].map(lead_profile_mapping)
    df["Tags"] = df["Tags"].map(tags_mapping)
    df["Specialization"] = df["Specialization"].map(specialization_mapping)
    df["Lead Quality"] = df["Lead Quality"].map(lead_quality_mapping)
    df["What is your current occupation"] = df["What is your current occupation"].map(occupation_mapping)
    df["How did you hear about X Education"] = df["How did you hear about X Education"].map(hear_about_mapping)

    return df
df = apply_custom_mappings(df)
