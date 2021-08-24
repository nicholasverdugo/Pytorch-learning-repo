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
    std::cout << "memcpy 1" << std::endl;
    std::memcpy(placeholder.data_ptr(), img.data, 1024 * 1024 * sizeof(unsigned char)); //problem is happening here - fixed it by changing from int to unsigned char
    //inputs.push_back(torch::ones({ 1, 1, 1024, 1024 }));
    std::cout << "Pushing back" << std::endl;
    //add the placeholder's data (copied from Mat img) into the IValue vector named inputs
    inputs.push_back(placeholder);


    // Execute the model and turn its output into a tensor.
    std::cout << "creating Tensor_output" << std::endl;
    torch::Tensor output = module.forward(inputs).toTensor(); //changed from at::Tensor to torch::Tensor
    //std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
    std::cout << "Tensor Output Created" << std::endl;
    //create an IValue vector to store the modified image data
    std::vector<torch::jit::IValue> outputs;

    //add the modified image data to the vector
    outputs.push_back(output);

    //write the image data to a mat, so OpenCV can display  it
    std::cout << "Reading Image Output" << std::endl;
    cv::Mat out = imread(path, cv::IMREAD_GRAYSCALE);

    std::cout << "memcpy 2" << std::endl;
    std::memcpy(output.data_ptr(), out.data, 1024 * 1024 * sizeof(unsigned char) * output.numel());
    //Write the image to a file.
    std::cout << "writing processed image" << std::endl;
    imwrite("image_processed.png", out);
    
    std::cout << "displaying processed image" << std::endl;
    //Display the processed image
    imshow("Processed Image", out);
}