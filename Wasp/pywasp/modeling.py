from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression

from pyspark.ml.feature import StandardScaler,       \
                               VectorAssembler,      \
                               StringIndexer,        \
                               OneHotEncoderEstimator

#----------------------------------------------------------------------
def Featurizer(categorical):
    
    normalizer      = StandardScaler(inputCol  = 'numerical',
                                     outputCol = 'numerical_norm'
                                    )

    label_encoder   = [StringIndexer(inputCol  = col,
                                     outputCol = col + '_label',
                                     handleInvalid  = 'keep')\
                       for col in categorical]
  
    one_hot_encoder = OneHotEncoderEstimator(\
                       inputCols  = [c + '_label' for c in categorical],
                       outputCols = [c + '_encod' for c in categorical],
                       handleInvalid  = 'keep'
                      )    
    
    assemblerInputs = ['numerical_norm']\
                      + [c + "_encod" for c in categorical]
    assembler = VectorAssembler(inputCols = assemblerInputs, 
                                outputCol = "features")

    featurizer = Pipeline(stages =                                      \
                            [normalizer]      +                         \
                            label_encoder     +                         \
                            [one_hot_encoder] +                         \
                            [assembler]
                 )
    
    return featurizer

#----------------------------------------------------------------------
def Numerizer(numerical):
    
    numerizer = VectorAssembler(inputCols = numerical, 
                                outputCol = 'numerical',
                                handleInvalid = 'skip'
                               )
    return numerizer


#----------------------------------------------------------------------
def builder(numerical_feat, 
            categorical_feat,
            clf = LogisticRegression(featuresCol = 'features', 
                                     labelCol    = 'label', 
                                     maxIter     = 2000)
            ):
    
    numerizer  = Numerizer(numerical_feat)
    featurizer = Featurizer(categorical_feat)
    
    model = Pipeline(stages = [
                     numerizer,
                     featurizer,
                     clf
            ])

    
    return model
