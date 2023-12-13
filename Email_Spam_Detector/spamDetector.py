import pickle
import streamlit as st


model=pickle.load(open("spam.pkl", "rb"))
cv=pickle.load(open("Vectorizer.pkl", "rb"))


def main():
	st.title("EMAIL SPAM CLASSIFIER")
	st.subheader("build with streamlit and python")
	msg=st.text_input("Enter a text: ")
	if st.button("predict"):
		data=[msg]
		vect=cv.transform(data).toarray()
		prediction=model.predict(vect)
		result=prediction[0]
		if result==1:
			st.error("This is spam mail")
		else:
			st.success("This is a ham mail")

main()