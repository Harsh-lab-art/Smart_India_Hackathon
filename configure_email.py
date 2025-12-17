#!/usr/bin/env python3
"""
Email Configuration Helper
Run this script to easily configure email settings
"""

import os
import re

def validate_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def configure_email():
    """Interactive email configuration"""
    print("=" * 60)
    print("📧 Email Configuration Helper")
    print("=" * 60)
    print()
    
    # Get email
    while True:
        email = input("Enter your Gmail address: ").strip()
        if validate_email(email):
            break
        print("❌ Invalid email format. Please try again.")
    
    # Get app password
    print()
    print("📝 Generate App Password:")
    print("   1. Go to: https://myaccount.google.com/apppasswords")
    print("   2. Select 'Mail' and 'Other (Custom name)'")
    print("   3. Enter name: 'Crypto Detection'")
    print("   4. Copy the 16-character password")
    print()
    
    app_password = input("Enter your App Password (16 chars, spaces ok): ").strip()
    app_password = app_password.replace(" ", "")  # Remove spaces
    
    if len(app_password) != 16:
        print(f"⚠️  Warning: App password should be 16 characters (got {len(app_password)})")
        confirm = input("Continue anyway? (y/n): ").strip().lower()
        if confirm != 'y':
            print("❌ Configuration cancelled")
            return
    
    # Create .env file
    env_content = f"""# Email Configuration
EMAIL_USER={email}
EMAIL_PASSWORD={app_password}

# MongoDB Configuration
MONGODB_URI=mongodb+srv://harsh9760verma_db_user:n2H9I0USOXordI5B@cluster0.rffbk5i.mongodb.net/
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print()
    print("=" * 60)
    print("✅ Configuration saved to .env file")
    print("=" * 60)
    print()
    print("📋 Next steps:")
    print("   1. Install python-dotenv: pip install python-dotenv")
    print("   2. Update api_server.py to use environment variables")
    print("   3. Restart the server: python api_server.py")
    print("   4. Test password reset on /login.html")
    print()
    print("🔒 Security:")
    print("   - .env file created (keep it private)")
    print("   - Add .env to .gitignore")
    print("   - Never commit credentials to Git")
    print()
    
    # Show configuration
    print("📧 Your configuration:")
    print(f"   Email: {email}")
    print(f"   Password: {'*' * len(app_password)}")
    print()
    
    # Test option
    test = input("Would you like to test the email configuration? (y/n): ").strip().lower()
    if test == 'y':
        test_email_config(email, app_password)

def test_email_config(email, password):
    """Test email configuration"""
    print()
    print("🧪 Testing email configuration...")
    
    try:
        import smtplib
        from email.mime.text import MIMEText
        
        # Create test message
        msg = MIMEText("This is a test email from Crypto Detection System.")
        msg['Subject'] = "Test Email - Crypto Detection"
        msg['From'] = email
        msg['To'] = email
        
        # Send email
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(email, password)
        server.send_message(msg)
        server.quit()
        
        print("✅ Email sent successfully!")
        print(f"📬 Check your inbox: {email}")
        
    except Exception as e:
        print(f"❌ Email test failed: {e}")
        print()
        print("Common issues:")
        print("   - 2-Step Verification not enabled")
        print("   - App Password incorrect")
        print("   - Internet connection issue")
        print("   - Firewall blocking SMTP")

if __name__ == "__main__":
    try:
        configure_email()
    except KeyboardInterrupt:
        print("\n\n❌ Configuration cancelled")
    except Exception as e:
        print(f"\n❌ Error: {e}")
