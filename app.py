import streamlit as st
import numpy as np
import joblib


model = joblib.load('finalized_model.sav')

@st.cache()

def prediction(vintage, current_balance, average_monthly_balance_prevQ,
       average_monthly_balance_prevQ2,current_month_debit,
       previous_month_debit,previous_month_balance):
       inp=np.array([[vintage, current_balance, average_monthly_balance_prevQ,
       average_monthly_balance_prevQ2,current_month_debit,
       previous_month_debit,previous_month_balance]]).astype(np.float64)
       pred_proba = model.predict_proba(inp)
       return pred_proba
   
def main():
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:Green;padding:13px"> 
    <h1 style ="color:white;text-align:center;">Bank Customer Churn Prediction App</h1> 
    </div> 
    """
    
    st.set_page_config("Churn Prediction")
    
    st.markdown(html_temp,unsafe_allow_html=True)
    
    vintage=st.text_input('Number of days of your account in the bank')
    current_balance=st.text_input('Current Balance')
    average_monthly_balance_prevQ=st.text_input('Average Monthly Balance in Previous Quarter')
    average_monthly_balance_prevQ2=st.text_input('Average Monthly Balance in Previous to Previous Quarter')
    current_month_debit = st.text_input('Current Month Debit')
    previous_month_debit=st.text_input('Previous Month Debit')
    previous_month_balance=st.text_input('Previous Month Balance')
    
    if st.button('Predict'):
        pred_proba=prediction(vintage, current_balance, average_monthly_balance_prevQ,
        average_monthly_balance_prevQ2,current_month_debit,
        previous_month_debit,previous_month_balance)
        pred=pred_proba.argmax(axis=-1)
        if pred==0:
               result=pred_proba[:,0]*100
               st.success("Confidence of customer staying in the bank - {} %".format(result))
        else:
               result=pred_proba[:,1]*100
               st.error("Confidence of customer not staying in the bank - {} %".format(result))
 
if __name__ == '__main__':
    main()