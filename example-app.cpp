#include <torch/script.h> // One-stop header.
#include <opencv2/core.hpp>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
    if (argc != 3) {
        std::cerr << "usage: example-app <path-to-exported-script-module> <path-to-tif-file>\n";
        return -1;
    }

    torch::jit::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(argv[1]);
    }
    catch (const c10::Error& e) {
        std::cerr << e.what();
        std::cerr << "error loading the model\n";
        return -1;
    }

    std::cout << "executing...\n";

    //display the image before running it through the model
    std::string path = argv[2];
    cv::Mat img = imread(path, cv::IMREAD_GRAYSCALE);

    //Check if the image file contains data.
    if (img.empty())
    {
        std::cout << "Could not read the image: " << path << std::endl;
        return 1;
    }

    //Display the image.
    imshow("Pre-pass image", img);

    int k = cv::waitKey(0); // Wait for a keystroke in the window to continue
    //if (k == 's')
    //{
    //    //Write the image to a file.
    //    imwrite("image_processed.png", img);
    //}
    
    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    //creates a placeholder tensor with the correct image size, and copies the image's data into it once it is created
    torch::Tensor placeholder(torch::zeros({ 1, 1, 1024, 1024 }));

    std::cout << "converting Mat to Tensor" << std::endl;
    std::memcpy(placeholder.data_ptr(), img.data, 1024 * 1024 * sizeof(unsigned char));

    //add the placeholder's data (copied from Mat img) into the IValue vector named inputs
    inputs.push_back(placeholder);


    // Execute the model and turn its output into a tensor.
    std::cout << "forward passing..." << std::endl;
    at::Tensor output = module.forward(inputs).toTensor(); //changed from at::Tensor to torch::Tensor
    std::cout << "forward pass complete." << std::endl;

    //Converting the tensor to a Mat so it can be displayed with cv
    //not sure if these next two lines are needed...final image weirdness might be due to the neural net. not sure though.
    //std::cout << "switching rows and cols..." << std::endl;
    //output = output.permute({2,3,1,0});
    std::cout << "converting the outputted Tensor to a Mat to be displayed..." << std::endl;
    cv::Mat out(1024, 1024, CV_32SC1, output.data_ptr());

    //memcpy does not actually need to happen since Mat creation can take in a tensor data ptr, leaving this here for future reference though.
    //std::memcpy(out.data, output.data_ptr(), output.numel() * sizeof(unsigned char));

    //Write the image to a file.
    std::cout << "writing processed image..." << std::endl;
    imwrite("image_processed.png", out);
    
    //Display the processed image
    std::cout << "displaying processed image..." << std::endl;
    imshow("Processed Image", out);
    k = cv::waitKey(0);
}