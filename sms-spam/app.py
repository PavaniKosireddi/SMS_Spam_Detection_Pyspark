import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.functions import length

# Initialize Spark session
spark = SparkSession.builder.appName("NLP-NB-Streamlit").getOrCreate()

# Load dataset and prepare the pipeline and model
df = spark.read.csv("SMSSpamCollection", sep="\t", inferSchema=True)
df = df.withColumnRenamed("_c0", "class").withColumnRenamed("_c1", "text")
df = df.withColumn("length", length(df["text"]))

from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF, StringIndexer, VectorAssembler
from pyspark.ml.classification import NaiveBayes
from pyspark.ml import Pipeline

tokenizer = Tokenizer(inputCol="text", outputCol="token_text")
stop_word_remover = StopWordsRemover(inputCol="token_text", outputCol="stop_tokens")
count_vec = CountVectorizer(inputCol="stop_tokens", outputCol="c_vec")
idf = IDF(inputCol="c_vec", outputCol="tf_idf")
ham_spam_to_num = StringIndexer(inputCol="class", outputCol="label")
cleaned = VectorAssembler(inputCols=["tf_idf", "length"], outputCol="features")

data_prep_pipe = Pipeline(stages=[ham_spam_to_num, tokenizer, stop_word_remover, count_vec, idf, cleaned])
fitted_pipeline = data_prep_pipe.fit(df)
final_data = fitted_pipeline.transform(df).select("label", "features")

# Train Naive Bayes
train, test = final_data.randomSplit([0.7, 0.3])
nb = NaiveBayes()
spam_detector = nb.fit(train)

# Streamlit App
st.title("📩 SMS Spam Detection")
st.write("Enter an SMS message below to predict whether it's Spam or Not Spam.")

user_input = st.text_area("Enter your message:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        # Create DataFrame for input
        user_df = spark.createDataFrame([(user_input,)], ["text"])
        user_df = user_df.withColumn("length", length(user_df["text"]))

        # Transform using pipeline
        user_cleaned = fitted_pipeline.transform(user_df).select("features")

        # Predict
        user_prediction = spam_detector.transform(user_cleaned)
        prediction_result = user_prediction.select("prediction").collect()[0][0]

        if prediction_result == 0.0:
            st.success("✅ Prediction: HAM (Not Spam)")
        else:
            st.error("🚫 Prediction: SPAM")
