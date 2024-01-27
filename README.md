# Serverless Neural Network

Serverless Neural Network is a lightweight framework designed for deploying trainable neural networks on serverless platforms, specifically tailored for AWS Lambda. Traditional neural network libraries like PyTorch, TensorFlow, and JAX offer extensive functionality, but they come with the drawback of being heavy and resource-intensive. This framework focuses on simplicity and efficiency, allowing you to deploy a lightweight neural network directly to AWS Lambda without the need for dedicated servers or complex container lambdas.

## Features

**Lightweight Deployment:** Deploy simple neural network on AWS Lambda without the need for dedicated servers or heavyweight containers.

**Data Handling:** Easily save datapoints, train a network on the stored data, and persist the trained model to a database.

**Inference Support:** Perform inferences using the trained network, making it suitable for real-time applications.

**Database Integration:** Uses Pymongo over Object-Document Mappers (ODMs) for efficient database operations, reducing query times and improving overall performance.

**Optimized Size:** The framework is uses bare minimum external libraries to keep the package size small (24 MB).


## Prerequisites

1. **Install Serverless Plugins:**

Ensure you have the `serverless` framework and `serverless-python-requirements` plugin installed. You can install them using the following commands:

```
npm install -g serverless
serverless plugin install -n serverless-python-requirements
```
For more details, refer to [serverless-python-requirements plugin](https://www.serverless.com/plugins/serverless-python-requirements).

2. **Setup AWS CLI:**

Make sure you have set up the AWS CLI with the necessary credentials. For detailed instructions, visit [AWS CLI Configuration Guide](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html).


## Setup and Deployment

1. **Clone the Repository:**

```
git clone https://github.com/yourusername/serverless-neural-network.git
cd serverless-neural-network
```

2. **Create .env file:**

Create a .env file using the provided .env.sample file. Replace the placeholder URL_DB with your actual database connection string.

3. **Populate Test Data (Optional):**

Run the following command to populate the database with test data:

```
pip install -r requirements.txt
python src/init_testdata.py
```

This script initializes the database with test data, ensuring your setup is complete for testing and development.


4. **Deploy to AWS Lambda:**

Execute the following command to deploy the Serverless Neural Network:

```
serverless deploy
```

This command will package and deploy the framework to AWS Lambda, with HTTP endpoints triggers.


## Usage

After deployment, you will have three endpoints deployed at an API Gateway host (`xxx.execute-api.rrr.amazonaws.com`):

* **/datapoint:** Use the following curl command to add data to your data points. This endpoint will add the datapoint to the database:

```
curl -X POST https://xxx.execute-api.rrr.amazonaws.com/datapoint \
       -H "Content-Type: application/json" \
       -d '{"x": [2.0, 3.0, -1.0], "y": 1}'
```

* **/train:** Use the following curl command to train your network. This endpoint will train on the datapoints stored in the database:

```
curl -X POST https://xxx.execute-api.rrr.amazonaws.com/train \
     -H "Content-Type: application/json"
```

* **/infer:** Use the following curl command to get inferences for your given input:

```
curl -X POST https://xxx.execute-api.rrr.amazonaws.com/infer \
     -H "Content-Type: application/json" \
     -d '{"x": [2.0, 3.0, -1.0]}'
```

*Optional:* You can also pass a network_id to infer with a given version of the neural network:

```
curl -X POST https://xxx.execute-api.rrr.amazonaws.com/infer -H 'Content-Type: application/json'-d '{"x": [2.0, 3.0, -1.0], "network_id": "your_network_id"}'
```

* **Testing Locally:** To test the handlers locally, navigate to `src/controllers/handler.py` and run the file. For testing the neural network, go to `src/entities/neuralnetwork.py` and run the file. Ensure you have the necessary dependencies installed before running the files locally. You can check the `requirements.txt` file for the required packages.

## Known Issues

⚠️ The neural network in this framework is being optimized and is currently highly untested. It may not perform reliably in all scenarios. Please exercise caution if you plan on using it for production.

* Loss Explosion:

The loss may explode in some cases due to suboptimal initialization

* Batch Normalization:

Batch normalization is not yet implemented, which may affect the model's performance. 

Work is in progress to add support to both the issues.



## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Forking and Usage

While the future support of this project is uncertain, anyone is welcome to fork and use this framework to build their own codebase. Feel free to customize and extend it based on your requirements.

**Note:** The framework is currently in active development, and additional features and improvements are planned. Stay tuned for updates!