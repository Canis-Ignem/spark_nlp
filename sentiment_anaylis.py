import findspark
findspark.init()
import pyspark as ps
from pyspark.sql import SQLContext


from pyspark.ml.feature import HashingTF, IDF, Tokenizer, CountVectorizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes, LinearSVC, MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline


sc = ps.SparkContext.getOrCreate()
sqlContext = SQLContext(sc)

df_train = sqlContext.read.format('com.databricks.spark.csv').options(header = False, inferschema = True, sep = ";").load("sentiment_data/train.csv")
df_train= df_train.withColumnRenamed('_c0','description').withColumnRenamed('_c1','feeling')
df_train.show(10)


df_test = sqlContext.read.format('com.databricks.spark.csv').options(header = False, inferschema = True, sep = ";").load("sentiment_data/val.csv")
df_test= df_test.withColumnRenamed('_c0','description').withColumnRenamed('_c1','feeling')
df_test.show(10)

print(df_train.count())
df_train.dropna()
df_train.count()


print(df_test.count())
df_test.dropna()
df_test.count()

label_stringIdx = StringIndexer(inputCol = 'feeling', outputCol = 'label')
tokenizer = Tokenizer(inputCol = 'description', outputCol = 'tokens')
hashingtf = HashingTF(numFeatures= 2**16, inputCol = "tokens", outputCol = 'tf')
idf = IDF(inputCol = "tf", outputCol = 'features', minDocFreq=3)

pipeline = Pipeline(stages = [label_stringIdx, tokenizer, hashingtf, idf ])

pipeline = pipeline.fit(df_train)
df_train = pipeline.transform(df_train)
df_test = pipeline.transform(df_test)

df_train.show(10)

df_test.show(10)

nb = NaiveBayes(smoothing=1.0, modelType="multinomial")

nb = nb.fit(df_train)

predictions = nb.transform(df_test)

criterion = MulticlassClassificationEvaluator()
acc = criterion.evaluate(predictions)

print(acc)