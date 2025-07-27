# Cardiovascular Health Mobile Application

## Overview

The Cardiovascular Health mobile application is a simple, privacy-focused tool for assessing cardiovascular risk and receiving health recommendations. The app does not require user accounts, credentials, or store personal data. It calls only two APIs: one for risk score calculation and one for health guides.

## Target Audience

- Adults interested in a quick cardiovascular risk check
- Anyone seeking health recommendations without sharing personal data

## App Features and Functionality

### Welcome Screen

- Displays information about the app's privacy: "We do not expose your data, and we do not ask for credentials. Your health information is never stored."

### Input Screen

- Users manually enter basic health metrics required for risk assessment:
   - Age
   - Gender
   - Height
   - Weight
   - Blood pressure
   - Cholesterol level
   - Glucose level
   - Smoking status
   - Alcohol consumption
   - Physical activity level

### Feedback Screen

- The app calls the `/predict` API to calculate the risk score.
- The app calls the `/guides` API to retrieve health recommendations based on the risk score.
- Displays the risk score (0-1) and a simple explanation (Low, Moderate, High risk).
- Shows actionable health recommendations.

## User Flow

1. **Welcome Screen**
   - App installation
   - Display privacy information

2. **Input Screen**
   - User enters health metrics

3. **Feedback Screen**
   - App calls two APIs: `/predict` for risk score, `/guides` for recommendations
   - Displays risk score and health recommendations

## API Integration

The app uses only two APIs:

1. `/predict` - Calculates cardiovascular risk score based on user input.
 request:ageCollapse allinteger[7300, 36500]
Age in days (7300-36500)

genderCollapse allinteger[1, 2]
Gender (1=female, 2=male)

heightCollapse allinteger[120, 220]
Height in cm (120-220)

weightCollapse allnumber[40, 200]
Weight in kg (40-200)

ap_hiCollapse allinteger[80, 240]
Systolic blood pressure (80-240 mmHg)

ap_loCollapse allinteger[40, 160]
Diastolic blood pressure (40-160 mmHg)

cholesterolCollapse allinteger[1, 3]
Cholesterol level (1=normal, 2=above normal, 3=well above normal)

glucCollapse allinteger[1, 3]
Glucose level (1=normal, 2=above normal, 3=well above normal)

smokeCollapse allinteger[0, 1]
Smoking status (0=non-smoker, 1=smoker)

alcoCollapse allinteger[0, 1]
Alcohol intake (0=doesn't drink, 1=drinks)

activeCollapse allinteger[0, 1]
Physical activity (0=not active, 1=active)
2. `/guides` - Returns health recommendations based on the risk score.
 request:
 {
  "risk_score": number
}

## Health Guide System

Recommendations are provided based on risk category:

- **Low Risk (0-0.3):** Maintenance and preventive tips
- **Moderate Risk (0.3-0.7):** Targeted lifestyle changes
- **High Risk (0.7-1.0):** Urgent health actions and advice to consult a provider

## Technical Implementation

- Cross-platform development (iOS and Android)
- Secure API communication (no authentication required)
- No local data storage or cloud sync
- Simple, privacy-first design

## Future Enhancements

- Optional integration with health devices (if privacy can be maintained)
- Expanded recommendations and educational content


response for api are here: 
 {"risk_score": 0.42,}

 / quide

 {
  "risk_category": "Moderate",
  "risk_score": 0.42,
  "general_advice": "Your cardiovascular risk is moderate. While not immediately concerning, you should take steps to improve your heart health.",
  "recommendations": [
    {
      "title": "Regular Blood Pressure Monitoring",
      "description": "Check your blood pressure at least once a month and keep a log of readings.",
      "priority": 2
    },
    {
      "title": "Dietary Adjustments",
      "description": "Reduce sodium intake and increase consumption of fruits, vegetables, and whole grains.",
      "priority": 1
    },
    {
      "title": "Regular Exercise",
      "description": "Aim for at least 150 minutes of moderate-intensity exercise per week.",
      "priority": 1
    }
  ]
}