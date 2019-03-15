using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using static System.Math;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.IntegralTransforms;
using MathNet.Numerics.Data.Text;
using CNTK;
using engine;



namespace engine
{

    public static class AuxFunctions{

        // convert one .csv file to .ctf and process it to prepare for NN
        public static void CreateOneExample(string DataFolder = "C:\\Users\\ITRI\\Documents\\Programming\\Csharp\\tooldataNoupload", string DataTimeFile = "F01ac001X_circle.csv", 
            string outFile = "C:\\Users\\ITRI\\Documents\\Programming\\Csharp\\Learning_CNTK\\Data\\procFreq.ctf",
            int freqSamples = 131072, int bins = 6000, int binDiv = 10)
        {
            

            // load and process data
            Matrix<float> timeDmnData = DelimitedReader.Read<float>(Path.Combine(DataFolder, DataTimeFile),false,",");

            Vector<float> timeDmnDataV = timeDmnData.Column(0);
            
            Complex32[] freqDataC = new Complex32[freqSamples];

            for (int i = 0; i < freqSamples; i++) // toDo: allocate space better?
            {
                freqDataC[i] = new Complex32(timeDmnDataV[i], 0);
            }

            // This method outputs data very similar to matlab's shape, but very different amplitude magnitude
            Fourier.Forward(freqDataC);

            float[] processedData = new float[bins];
            for (int i = 0; i < bins; i++)
            {
                // get real and abs, then mean it
                //Abs(freqDataC[i].Real);
                // to mean it, I need to get ((i-1)*binDiv):(i*binDiv)) range of values and divide them by binDiv
                float mean = 0;

                for (int k = 0; k < binDiv; k++)
                {
                    mean = mean + Abs(freqDataC[(i * binDiv + k)].Real);
                }

                processedData[i] = mean / binDiv;

            }


            // CNTK file format, example file with worn condition
            // |labels 1 0 0 |features 0.001 1.3 3.009 \n
            var sw = File.CreateText(outFile);

            sw.Write("|labels 0 1 |features "); // toDo: automagically determine labels
            foreach (float fdpoint in processedData)
            {
                sw.Write(fdpoint * 1000); sw.Write(" "); // because the amplitude is different, I multiplied the values to match the training data for now
                // toDo: remove the multiplier and train the data on these samples
            }
            sw.Write("\n");

            sw.Flush();
            sw.Close();

        }


        // use the one .ctf file from above function in a trained model
        public static void TestData(string dataPath_testfile = "C:\\Users\\ITRI\\Documents\\Programming\\Csharp\\Learning_CNTK\\Data\\procFreq.ctf")
        {
            var device = DeviceDescriptor.CPUDevice;


            int numOutputClasses = 2; // need to know these from the start
            uint inputDim = 6000;   // also these
            uint batchSize = 1; // not sure how to make this increase to 100% of the file, for now

            string dataPath_model = "C:\\Users\\ITRI\\Documents\\Programming\\Csharp\\Learning_CNTK\\Data\\mModelZ1.dnn";
            //string dataPath_train = "C:\\Users\\ITRI\\Documents\\Programming\\Csharp\\Learning_CNTK\\Data\\YXFFData6001Train.txt";


            // load saved model
            Function model = Function.Load(dataPath_model, device);


            // the model output needs to be processed still
            // out = C.softmax(z)
            var modelOut = CNTKLib.Softmax(model); // softmax is just an extra function that wasn't put inside the model. 
            // toDo: decide whether to put the softmax into the model


            // these variables are used to map some functions together
            //var feature_fromModel = modelOut.Arguments[0];
            var label_fromModel = modelOut.Output;

            string featureStreamName = "features"; // these are the names inside the .ctf file
            string labelsStreamName = "labels";


            // preparing to open the file stream
            var streamConfig = new StreamConfigurationVector{
                    new StreamConfiguration(featureStreamName, inputDim),
                    new StreamConfiguration(labelsStreamName, numOutputClasses)
            };


            // deserialiser is used to read the data, there are a few inbuild ones
            var deserializerConfig = CNTKLib.CTFDeserializer(dataPath_testfile, streamConfig);


            MinibatchSourceConfig MBconfig_train = new MinibatchSourceConfig(new List<CNTKDictionary> { deserializerConfig })
            {
                MaxSweeps = 1000, // toDo: understand these parameters better
                randomizationWindowInChunks = 0,
                randomizationWindowInSamples = 100000,
            };


            // this is finally the stream controller
            var MBsource = CNTKLib.CreateCompositeMinibatchSource(MBconfig_train);
            // these are used to extract data from stream
            var featureStreamInfo = MBsource.StreamInfo(featureStreamName);
            var labelStreamInfo = MBsource.StreamInfo(labelsStreamName);


            // this gets the 'next batch' of data
            var NextBatch = MBsource.GetNextMinibatch(batchSize, device);


            // also some connecting helpers
            var MBdensefeature = NextBatch[featureStreamInfo].data;
            var MBdenseLabel = NextBatch[labelStreamInfo].data.GetDenseData<float>(label_fromModel); // this one actually contains label data from file

            
            //define input and output variable and connecting to the stream configuration
            var feature = Variable.InputVariable(new NDShape(1, inputDim), DataType.Float, featureStreamName);
            var label = Variable.InputVariable(new NDShape(1, numOutputClasses), DataType.Float, labelsStreamName);

            
            //input, features structure mapping
            Variable inputVar = modelOut.Arguments.Single();

            var inputDataMap = new Dictionary<Variable, Value>();
            inputDataMap.Add(inputVar, MBdensefeature);


            //output, labels structure mapping
            var outputDataMap = new Dictionary<Variable, Value>();
            Variable outputVar = modelOut.Output;
            outputDataMap.Add(outputVar, null);




            // evaluate with loaded data
            modelOut.Evaluate(inputDataMap, outputDataMap, device); // the maps and data inside are modified

            // this is predicted data
            var outputData = outputDataMap[outputVar].GetDenseData<float>(outputVar);




            // making a list of predicted data, changing it into binary prediction representation
            var PredictedLabels = outputData.Select((IList<float> l) => l.IndexOf(l.Max())).ToList();

            // getting list of actual values from file in same format
            IList<int> expectedLabels = MBdenseLabel.Select(l => l.IndexOf(1.0F)).ToList();

            // number of missed predictions
            int misMatches = PredictedLabels.Zip(expectedLabels, (a, b) => a.Equals(b) ? 0 : 1).Sum();

            // simply amount of samples/labels
            int labelsLength = PredictedLabels.Count;

            // quick helper function for output
            string correctness(bool comparison)
            {
                if (comparison) return "Correct prediction";
                else return "Incorrect prediction";
            }

            // output predictions in the console
            for (int i = 0; i < labelsLength; i++)
            {
                Console.WriteLine($"{i + 1}.\tPredicted value:  {PredictedLabels[i]};\tExpected value:  {expectedLabels[i]};\t{correctness(PredictedLabels[i] == expectedLabels[i])}.");

            }

            Console.WriteLine($"Validating Model: Total Samples = {batchSize}, Misclassify Count = {misMatches}.");


            
            Console.WriteLine("Success");
        }

        
        
        // converts specifically to needed format, same as 1 example, but implement proper classification
        public static void Convert2CTF(int outClasses, int thisClass, string dataPath_in, string DataFolder_in = @"C:\Users\ITRI\Documents\Programming\Csharp\tooldataNoupload\circle", 
            string DataFolder_out = @"C:\Users\ITRI\Documents\Programming\Csharp\tooldataNoupload\CTFdata", string dataPath_out = "freqD.ctf")
        {

            int freqSamples = 131072;
            int bins = 6000;
            int binDiv = 10;
            string inFile = Path.Combine(DataFolder_in, dataPath_in);

            if (File.Exists(inFile)) Console.WriteLine("Found {0}, processing...", dataPath_in);
            else
            {
                Console.WriteLine("{0} not found, returning...", dataPath_in);
                return;
            }
            
            
            string outFile = Path.Combine(DataFolder_out, dataPath_out);


            // load and process data
            // read .csv
            Matrix<float> timeDmnData = DelimitedReader.Read<float>(inFile, false, ",");

            // toDo: optimise all these conversions, are they needed?
            Vector<float> timeDmnDataV = timeDmnData.Column(0);

            Complex32[] freqDataC = new Complex32[freqSamples];

            for (int i = 0; i < freqSamples; i++) // toDo: allocate space better?
            {
                freqDataC[i] = new Complex32(timeDmnDataV[i], 0);
            }

            // This method outputs data very similar to matlab's shape, but very different amplitude magnitude
            Fourier.Forward(freqDataC);
            
            float[] processedData = new float[bins];
            for (int i = 0; i < bins; i++)
            {
                // get real and abs, then mean it
                //Abs(freqDataC[i].Real);
                // to mean it, I need to get ((i-1)*binDiv):(i*binDiv)) range of values and divide them by binDiv
                float mean = 0;

                for (int k = 0; k < binDiv; k++)
                {
                    mean = mean + Abs(freqDataC[(i * binDiv + k)].Real);
                }

                processedData[i] = mean / binDiv;

            }

            int[] labels2write = new int[outClasses];
            for(int i =0; i<outClasses; i++)
            {
                labels2write[i] = 0;
                if (i == thisClass) labels2write[i] = 1;
            }

            // CNTK file format, example file with worn condition
            // |labels 1 0 0 |features 0.001 1.3 3.009 \n
            if (!File.Exists(outFile))
            {
                Console.WriteLine("{0} file doesn't exist, creating...", dataPath_out);
                var sw = File.CreateText(outFile);

                sw.Write("|labels ");
                foreach (int l in labels2write)
                {
                    sw.Write(l); sw.Write(" ");
                }
                sw.Write("|features ");
                foreach (float fdpoint in processedData)
                {
                    sw.Write(fdpoint); sw.Write(" ");
                }
                sw.Write("\n");

                sw.Flush();
                sw.Close();
            }
            else if (File.Exists(outFile))
            {
                Console.WriteLine("Found {0} file, appending...", dataPath_out);
                var sw1 = new StreamWriter(outFile, true);

                sw1.Write("|labels ");
                foreach (int l in labels2write)
                {
                    sw1.Write(l); sw1.Write(" ");
                }
                sw1.Write("|features ");
                foreach (float fdpoint in processedData)
                {
                    sw1.Write(fdpoint); sw1.Write(" ");
                }
                sw1.Write("\n");

                sw1.Flush();
                sw1.Close();
            }
            


        }


        // selects specific .csv files in a folder to convert to .ctf using the above function
        // I've excluded Z accelerometer values and I collated the rest of the data together
        public static void Files2Convert2CTF(int outClasses = 3, int maxSharpness = 21, 
            string DataFolder = @"C:\Users\ITRI\Documents\Programming\Csharp\tooldataNoupload\")
        {

            
            int classDivHelper = maxSharpness / outClasses;

            // get list of files in the directory
            DirectoryInfo rawDataFolder = new DirectoryInfo(DataFolder);
            FileInfo[] rawDataFiles = rawDataFolder.GetFiles("*.csv", SearchOption.AllDirectories);

            foreach (FileInfo file in rawDataFiles)
            {
                if ((file.Name[8] != 'Z') && ((file.Name[2] != '1') && (file.Name[2] != '3')) && (file.Name[10] != 'l')) //&& (file.Name[10] != 'l')
                {
                    string sharpStr = file.Name.Substring(5, 3);
                    int sharpInd = 0;
                    int thisClass = 0;
                    if (Int32.TryParse(sharpStr, out sharpInd))
                    {
                        thisClass = (sharpInd) / classDivHelper;
                    }

                    AuxFunctions.Convert2CTF(outClasses, thisClass, file.Name, DataFolder, dataPath_out: "freqD3CS_train.ctf");

                    //Console.WriteLine(file.Name);
                }
            }
        }
















        public static void Helloworld()
        {
            Console.WriteLine("\n______________\n\nhello this is dog\n______________\n\n");
        }


    }

    public static class UseCNTK
    {


        // the following few functions chain the CNTK.Function class together to create a NN
        public enum Activation
        {
            None,
            ReLU,
            Sigmoid,
            Tanh
        }
        static Function ApplyActivationFunction(Function layer, Activation actFun)
        {
            switch (actFun)
            {
                default:
                case Activation.None:
                    return layer;
                case Activation.ReLU:
                    return CNTKLib.ReLU(layer);
                case Activation.Sigmoid:
                    return CNTKLib.Sigmoid(layer);
                case Activation.Tanh:
                    return CNTKLib.Tanh(layer);
            }
        }

        static Function SimpleLayer(Function input, int outputDim, DeviceDescriptor device)
        {
            //prepare default parameters values
            var glorotInit = CNTKLib.GlorotUniformInitializer(
                    CNTKLib.DefaultParamInitScale,
                    CNTKLib.SentinelValueForInferParamInitRank,
                    CNTKLib.SentinelValueForInferParamInitRank, 1);

            //create weight and bias vectors
            var var = (Variable)input;
            var shape = new int[] { outputDim, var.Shape[0] };
            var weightParam = new Parameter(shape, DataType.Float, glorotInit, device, "w");
            var biasParam = new Parameter(new NDShape(1, outputDim), 0, device, "b");

            //construct W * X + b matrix
            return CNTKLib.Times(weightParam, input) + biasParam;
        }

        public static Function CreateFFNN(Variable input, int hiddenLayerCount, int hiddenDim, int outputDim, Activation activation, string modelName, DeviceDescriptor device)
        {
            //First the parameters initialization must be performed
            var glorotInit = CNTKLib.GlorotUniformInitializer(
                    CNTKLib.DefaultParamInitScale,
                    CNTKLib.SentinelValueForInferParamInitRank,
                    CNTKLib.SentinelValueForInferParamInitRank, 1);

            //hidden layers creation
            //first hidden layer
            Function h = SimpleLayer(input, hiddenDim, device);
            h = ApplyActivationFunction(h, activation);
            //2,3, ... hidden layers
            for (int i = 1; i < hiddenLayerCount; i++)
            {
                h = SimpleLayer(h, hiddenDim, device);
                h = ApplyActivationFunction(h, activation);
            }
            //the last action is creation of the output layer
            var r = SimpleLayer(h, outputDim, device);
            r.SetName(modelName);
            return r;
        }
        // end of chain



    }

    public class Class1
    {
    }
}
