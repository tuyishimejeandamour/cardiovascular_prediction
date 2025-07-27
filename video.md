# Video Demo Script and Scenes

This document outlines the structure and talking points for yMy 5-minute video demo, covering the mobile app, API, and model performance as required.

---

## Scene 1: Introduction (10 seconds)
- Presenter on camera: "Hello, my name is [YMy Name]. In this demo, I will show you My Cardiovascular Disease Prediction mobile app and API in action."

---

## Scene 2: Mobile App Demo (60 seconds)
- Screen recording of the mobile app on a real device or emulator.
- Show the welcome screen, then inputting user data (age, gender, blood pressure, etc.).
- Narrate: "The app allows users to enter their health information. No login or personal data is stored."
- Show the prediction result and recommendations screen.
- Narrate: "After submitting, the app calls My deployed API and displays the risk score and personalized health advice."
- Briefly show the Flutter code where the API call is made (open `api_service.dart` or relevant provider).
- Narrate: "Here is the Flutter code that sends user data to the API and handles the response."

---

## Scene 3: API Demo with Swagger UI (60 seconds)
- Switch to browser, open the public Swagger UI endpoint.
- Narrate: "This is My public API, documented with Swagger UI."
- Show the `/predict` endpoint, fill in valid values, and execute.
- Narrate: "The API validates input ranges and types. For example, age must be between 20 and 100 years, blood pressure within physiological limits, etc."
- Show an invalid input and the error message.
- Show the `/guides` endpoint with a risk score and the returned recommendations.

---

## Scene 4: Model Performance & Selection (60 seconds)
- Switch to Jupyter notebook.
- Narrate: "We trained and compared three models: Linear Regression, Decision Tree, and Random Forest."
- Show the notebook cells with model training and evaluation.
- Narrate: "Here are the loss metrics: Mean Squared Error, R², and others. Random Forest performed best on My dataset."
- Briefly explain why the dataset characteristics influenced model choice (e.g., nonlinearity, feature importance).
- Show the code cell where the best model is saved for deployment.

---

## Scene 5: Closing (10 seconds)
- Presenter on camera: "Thank you for watching. This concludes the demo of My Cardiovascular Disease Prediction app and API."

---

**Tips:**
- Keep each section concise and focused.
- Make sure yMy face is visible in the intro and closing.
- Use screen recording for app, code, Swagger UI, and notebook sections.
- Do not exceed 5 minutes total.
