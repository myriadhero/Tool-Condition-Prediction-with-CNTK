using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using CNTK;
using engine;





namespace console
{
    class Program
    {
        static void Main(string[] args)
        {



            //string DataFolder_ctf = @"C:\Users\ITRI\Documents\Programming\Csharp\tooldataNoupload\CTFdata";
            //string DataFile_ctf = "freqDcircSqL.ctf";
            //string DataFile_ctfFP = Path.Combine(DataFolder_ctf, DataFile_ctf);

            int numOutputClasses = 3;
            int inputDim = 6000;
            int hidenLayerDim = 15;
            int numHiddenLayers = 2;
            


            //input_dim = 6000
            //out_classes = 3
            //num_hidden_layers = 2
            //hidden_layers_dim = 10

            //learning_rate = 0.1
            //num_mb_iter = 100

            string dataPath_train = @"C:\Users\ITRI\Documents\Programming\Csharp\tooldataNoupload\CTFdata\freqD3CS_train.ctf";
            string dataPath_test = @"C:\Users\ITRI\Documents\Programming\Csharp\tooldataNoupload\CTFdata\freqD3CS_test.ctf";

            uint batchSize_train = 72;
            uint batchSize_test = 48;

            string featureStreamName = "features";
            string labelsStreamName = "labels";

            //# Initialize the parameters for the trainer
            //minibatch_size_train = 72
            //num_samples_per_sweep = 72
            //num_sweeps_to_train_with = 100
            //num_minibatches_to_train = (num_samples_per_sweep * num_sweeps_to_train_with) / minibatch_size_train

            //epochs = 10

            //#log every
            //num_train_prog = int(num_sweeps_to_train_with / 10)


            AuxFunctions.Helloworld();
            var device = DeviceDescriptor.CPUDevice;

            //# input variables denoting the features and label data
            //Variable featureVariable = Variable.InputVariable(new int[] { inputDim }, DataType.Float);
            //Variable labelVariable = Variable.InputVariable(new int[] { numOutputClasses }, DataType.Float);

            var feature = Variable.InputVariable(new NDShape(1, inputDim), DataType.Float, featureStreamName);
            var label = Variable.InputVariable(new NDShape(1, numOutputClasses), DataType.Float, labelsStreamName);


            // make model
            //Build simple Feed Froward Neural Network model
            // var ffnn_model = CreateMLPClassifier(device, numOutputClasses, hidenLayerDim, feature, classifierName);
            var ffnn_model = UseCNTK.CreateFFNN(feature, numHiddenLayers, hidenLayerDim, numOutputClasses, UseCNTK.Activation.ReLU, "ToolCondition", device);

            //Loss and error functions definition
            var trainingLoss = CNTKLib.CrossEntropyWithSoftmax(new Variable(ffnn_model), label, "lossFunction");
            var classError = CNTKLib.ClassificationError(new Variable(ffnn_model), label, "classificationError");



            // setting up stream
            var streamConfig = new StreamConfigurationVector{
                    new StreamConfiguration(featureStreamName, inputDim),
                    new StreamConfiguration(labelsStreamName, numOutputClasses)
            };


            var deserializerConfig_train = CNTKLib.CTFDeserializer(dataPath_train, streamConfig);

            var deserializerConfig_test = CNTKLib.CTFDeserializer(dataPath_test, streamConfig);

            MinibatchSourceConfig MBconfig_train = new MinibatchSourceConfig(new List<CNTKDictionary> { deserializerConfig_train })
            {
                MaxSweeps = 1000,
                randomizationWindowInChunks = 0,
                randomizationWindowInSamples = 100000,
            };

            MinibatchSourceConfig MBconfig_test = new MinibatchSourceConfig(new List<CNTKDictionary> { deserializerConfig_test })
            {
                MaxSweeps = 1000,
                randomizationWindowInChunks = 0,
                randomizationWindowInSamples = 100000,
            };

            var MBsource_train = CNTK.CNTKLib.CreateCompositeMinibatchSource(MBconfig_train);

            var MBsource_test = CNTK.CNTKLib.CreateCompositeMinibatchSource(MBconfig_test);

            var featureStreamInfo_train = MBsource_train.StreamInfo(featureStreamName);
            var labelStreamInfo_train = MBsource_train.StreamInfo(labelsStreamName);

            var featureStreamInfo_test = MBsource_test.StreamInfo(featureStreamName);
            var labelStreamInfo_test = MBsource_test.StreamInfo(labelsStreamName);



            // set learning rate for the network
            var learningRatePerSample = new TrainingParameterScheduleDouble(0.001125, 1);

            //define learners for the NN model
            var learner = Learner.SGDLearner(ffnn_model.Parameters(), learningRatePerSample);

            //define trainer based on ffnn_model, loss and error functions , and SGD learner
            var trainer = Trainer.CreateTrainer(ffnn_model, trainingLoss, classError, new Learner[] { learner });


            // get a batch of data
            var nextBatch_train = MBsource_train.GetNextMinibatch(batchSize_train, device);
            

            //var label_fromModel = ffnn_model.Output;


            //var MBdensefeature_train = nextBatch_train[featureStreamInfo_train].data;
            //var MBdenseLabel_train = nextBatch_train[labelStreamInfo_train].data.GetDenseData<float>(label_fromModel);



            var arguments_train = new Dictionary<Variable, MinibatchData>()
            {
                { feature, nextBatch_train[featureStreamInfo_train] },
                { label, nextBatch_train[labelStreamInfo_train] }
            };

            

            

            


            var modelOut = CNTKLib.Softmax(ffnn_model);

            

            for (int i = 1; i<=500; i++)
            {




                trainer.TrainMinibatch(arguments_train, device);

                if(i%5 == 0)
                {
                    
                    var nextBatch_test = MBsource_test.GetNextMinibatch(batchSize_test, device);

                    //input
                    Variable inputVar = modelOut.Arguments.Single();
                    var inputDataMap_test = new Dictionary<Variable, Value>()
                    {
                            { inputVar, nextBatch_test[featureStreamInfo_test].data }
                    };


                    //output
                    Variable outputVar = modelOut.Output; ;
                    var outputDataMap_test = new Dictionary<Variable, Value>()
                    {
                            { outputVar, null }
                    };

                    
                    
                    // use test minibatch
                    modelOut.Evaluate(inputDataMap_test, outputDataMap_test, device);

                    var outputData = outputDataMap_test[outputVar].GetDenseData<float>(outputVar);
                    var predLabels = outputData.Select(l => l.IndexOf(l.Max())).ToList();

                    var labelData = nextBatch_test[labelStreamInfo_test].data.GetDenseData<float>(outputVar);
                    var labelsInData = labelData.Select(l => l.IndexOf(l.Max())).ToList();

                    int misMatches = predLabels.Zip(labelsInData, (a, b) => a.Equals(b) ? 0 : 1).Sum();

                    Console.WriteLine($"{i} Training accuracy {(1.0 - trainer.PreviousMinibatchEvaluationAverage()): 0.00};\tTest Data accuracy: {(1.0 - (float)misMatches / (float)batchSize_test) : 0.00} ({misMatches} mismatches).");
                }


            }

            Console.WriteLine("fin");

            // get data stream in
            // train model
            // validate model











            Console.WriteLine("fin.");












        }
    }
}
