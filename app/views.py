from django.shortcuts import render, get_object_or_404 
import os
import yaml

from django.http import JsonResponse
import os

from django.core.mail import EmailMessage

from django.contrib.auth.models import User
from django.contrib.auth import login, authenticate,logout
from django.shortcuts import redirect

from django.contrib.auth.tokens import default_token_generator
from django.utils.http import urlsafe_base64_encode
from django.utils.http import urlsafe_base64_decode
from django.utils.encoding import force_bytes
from django.utils.encoding import force_str

from django.template import loader
from django.core.mail import EmailMessage
from django.core.mail import send_mail
import random
from django.shortcuts import render
from django.http import HttpResponse


def home(request):
    # email_content = """
    #         Hello,

    #         We are excited to invite you to Optiqs Hub!

    #         Discover a world of possibilities, connect with like-minded individuals, and stay updated with the latest in our community.

    #         We look forward to having you onboard.

    #         Best regards,  
    #         The Optiqs Hub Team  
    #         """

    # email_subject = 'Optiqs Hub!'
    # from_email = 'optiqshub@gmail.com'
    # recipients = ["bakadir.oussama@gmail.com"]
    # for recipient in recipients:
    #     msg_me = EmailMessage(email_subject,email_content,from_email,[recipient])
    #     msg_me.content_subtype = 'html'  # Indicate that the email content is HTML
    #     try:
        
    #         msg_me.send()
    #         # Rest of your code after sending the email
    #     except Exception as e:
    #         print("Email sending error:", e)
    
    
    context = {
      
    }
    
    return render(request, 'app/home.html', context)

def get_categories_and_materials(base_path):
    categories = {}
    try:
        for category in os.listdir(base_path):
            category_path = os.path.join(base_path, category)
            if os.path.isdir(category_path):
                materials = {}
                for material in os.listdir(category_path):
                    material_path = os.path.join(category_path, material)
                    if os.path.isdir(material_path):
                        nk_files = []
                        nk_path = os.path.join(material_path, "nk")
                        if os.path.exists(nk_path):
                            nk_files = [
                                f for f in os.listdir(nk_path) if f.endswith(".yml")
                            ]
                        materials[material] = nk_files
                if materials:  # Only add to categories if materials are found
                    categories[category] = materials
    except Exception as e:
        print(f"An error occurred: {e}")
    return categories

def refractiveindex_data(request):
    base_path = "static/thinfilms"
    categories_and_materials = get_categories_and_materials(base_path)

    context = {
        'categories_and_materials': categories_and_materials,  # Pass the data to the template
        'base_url':base_path,
    }
    
    return render(request, 'app/refractiveindex_data.html', context)




