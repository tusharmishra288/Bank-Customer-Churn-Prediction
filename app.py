import streamlit as st
import tensorflow as tf
import numpy as np


model = tf.keras.models.load_model('my_model.h5')

@st.cache()

def prediction(vintage, current_balance, average_monthly_balance_prevQ,
       average_monthly_balance_prevQ2, current_month_debit,
       previous_month_balance):
       inp=np.array([[vintage, current_balance, average_monthly_balance_prevQ,
       average_monthly_balance_prevQ2, current_month_debit,
       previous_month_balance]]).astype(np.float64)
       pred = model.predict(inp)
       classes=model.predict_classes(inp)
       #result='{0:.{1}f}'.format(pred[0][0], 2)
       return float(pred) ,classes    
                
def main():
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:Green;padding:13px"> 
    <h1 style ="color:white;text-align:center;">Bank Customer Churn Prediction App</h1> 
    </div> 
    """
    
    st.set_page_config("Churn Prediction")
    
    st.markdown(html_temp,unsafe_allow_html=True)
    
    vintage=st.text_input('Number of days of your account')
    current_balance=st.text_input('Current Balance')
    average_monthly_balance_prevQ=st.text_input('Average Monthly Balance in Previous Quarter')
    average_monthly_balance_prevQ2=st.text_input('Average Monthly Balance in Previous to Previous Quarter')
    current_month_debit = st.text_input('Current Month Debit')
    previous_month_balance=st.text_input('Previous Month Balance')
    
    if st.button('Predict'):
        pred,classes=prediction(vintage, current_balance, average_monthly_balance_prevQ,
           average_monthly_balance_prevQ2, current_month_debit,
           previous_month_balance)
        if pred > 0.5:
               result=pred*100
               st.error("Confidence of customer not staying in the bank - {} %".format(result))
        else:
               result=1-pred
               result=result*100
               st.success("Confidence of customer staying in the bank - {} %".format(result))
 
if __name__ == '__main__':
    main()