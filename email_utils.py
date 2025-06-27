import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import os
from datetime import datetime


def format_email_for_delivery(html_content):
    """
    Format HTML email content to improve deliverability and prevent emails from going to spam.

    Args:
        html_content (str): The HTML content of the email

    Returns:
        str: Formatted HTML content with improved deliverability
    """
    # Add proper HTML structure if not present
    if not html_content.strip().startswith('<!DOCTYPE html>'):
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Accident Severity Detection</title>
        </head>
        <body>
            {html_content}

            <div style="margin-top: 20px; padding-top: 10px; border-top: 1px solid #eee; font-size: 12px; color: #666;">
                <p>This email was sent from Accident Severity Detection System.</p>
                <p>If you received this email by mistake, please contact support.</p>
            </div>
        </body>
        </html>
        """

    # Add unsubscribe link to comply with anti-spam regulations
    if "<unsubscribe>" not in html_content.lower():
        html_content = html_content.replace("</body>",
                                            """
                                            <div style="margin-top: 20px; font-size: 11px; color: #999;">
                                                <p>To unsubscribe from these alerts, please <a href="https://yourdomain.com/unsubscribe" style="color: #999;">click here</a>.</p>
                                            </div>
                                            </body>
                                            """)

    return html_content


def send_alert_email(to_email, subject, image_path, severity_level, location):
    """
    Send an email alert with the accident image attached

    Args:
        to_email (str): Recipient email address
        subject (str): Email subject
        image_path (str): Path to the image file or GridFS file ID
        severity_level (str): Severity level of the accident
        location (str): Location where the accident was detected

    Returns:
        tuple: (success, message)
    """
    print(
        f"send_alert_email called with: to={to_email}, subject={subject}, image={image_path}, severity={severity_level}")

    try:
        from_email = "projectmailm@gmail.com"
        password = "qmgn xecl bkqv musr"  # App password for Gmail

        # Create message
        msg = MIMEMultipart()
        msg['From'] = from_email
        msg['To'] = to_email
        msg['Subject'] = subject

        # Customize colors and urgency based on severity
        if severity_level == "Severe":
            header_color = "#f44336"  # Red
            header_text = "⚠️ URGENT: Severe Accident Alert ⚠️"
            alert_text = "A SEVERE accident has been detected that requires immediate attention!"
            severity_color = "#f44336"  # Red
        elif severity_level == "Moderate":
            header_color = "#ff9800"  # Orange
            header_text = "⚠️ Moderate Accident Alert ⚠️"
            alert_text = "A MODERATE accident has been detected that requires attention."
            severity_color = "#ff9800"  # Orange
        elif severity_level == "Minor":
            header_color = "#ffeb3b"  # Yellow
            header_text = "Minor Accident Alert"
            alert_text = "A MINOR accident has been detected."
            severity_color = "#ff9800"  # Yellow
        else:
            header_color = "#2196f3"  # Blue
            header_text = "Image Analysis Results"
            alert_text = "Your image has been analyzed by the Accident Severity Detection System."
            severity_color = "#2196f3"  # Blue

        # Create HTML content with styling based on severity
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Accident Analysis Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; margin: 0; padding: 0; }}
                .container {{ max-width: 600px; margin: 0 auto; }}
                .header {{ background-color: {header_color}; color: white; padding: 15px 20px; text-align: center; }}
                .content {{ padding: 20px; background-color: #f9f9f9; }}
                .footer {{ font-size: 12px; color: #777; text-align: center; margin-top: 20px; padding: 10px; background-color: #f1f1f1; }}
                .severity {{ font-weight: bold; color: {severity_color}; font-size: 18px; }}
                .info-row {{ margin-bottom: 15px; }}
                .info-label {{ font-weight: bold; }}
                .alert-box {{ background-color: #f8f8f8; border-left: 4px solid {header_color}; padding: 15px; margin: 15px 0; }}
                .btn {{ display: inline-block; background-color: {header_color}; color: white; padding: 10px 20px; text-decoration: none; border-radius: 4px; font-weight: bold; margin-top: 10px; }}
                .btn:hover {{ background-color: #333; }}
                img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; margin: 15px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h2>{header_text}</h2>
                </div>
                <div class="content">
                    <div class="alert-box">
                        <p><strong>{alert_text}</strong></p>
                    </div>

                    <div class="info-row">
                        <span class="info-label">Severity Level:</span> 
                        <span class="severity">{severity_level.upper()}</span>
                    </div>

                    <div class="info-row">
                        <span class="info-label">Detection Time:</span> 
                        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    </div>

                    <div class="info-row">
                        <span class="info-label">Location/Source:</span> 
                        {location}
                    </div>

                    <p>Please see the attached image showing the analysis results:</p>

                    <p>This is an automated alert from the Accident Severity Detection System. Please take appropriate action based on the severity level.</p>
                </div>
                <div class="footer">
                    <p>© {datetime.now().year} Accident Severity Detection System</p>
                    <p>This is an automated message. Please do not reply to this email.</p>
                </div>
            </div>
        </body>
        </html>
        """

        # Attach HTML content
        msg.attach(MIMEText(html_content, 'html'))

        # Attach image if it exists
        if image_path and os.path.isfile(image_path):
            try:
                print(f"Attaching image from path: {image_path}")
                with open(image_path, 'rb') as attachment:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(attachment.read())
                    encoders.encode_base64(part)
                    filename = os.path.basename(image_path)
                    part.add_header('Content-Disposition', f'attachment; filename="{filename}"')
                    part.add_header('Content-ID', f'<{filename}>')
                    msg.attach(part)
                    print(f"Successfully attached image: {image_path}")
            except Exception as e:
                print(f"Error attaching image: {str(e)}")
                return False, f"Error attaching image: {str(e)}"
        else:
            print(f"Image path not found or invalid: {image_path}")

        # Connect to SMTP server
        try:
            print("Connecting to SMTP server...")
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            print("Logging in to email account...")
            server.login(from_email, password)

            # Send email
            print(f"Sending email to {to_email}...")
            server.send_message(msg)
            server.quit()

            print("Email sent successfully")
            return True, "Email sent successfully"
        except Exception as e:
            print(f"SMTP error: {str(e)}")
            return False, f"SMTP error: {str(e)}"

    except Exception as e:
        print(f"General error in send_alert_email: {str(e)}")
        return False, str(e)
