#include "Scene.h"
#include "Renderer.h"
#ifdef ENABLE_VISUALIZER
#include "Debug_Visualizer.h"
#endif


Scene* scene;
Renderer* renderer;

int main(){

	//Initialize Scene
	string config = "Input data\\lucy.obj";
	scene = new Scene(config);

	//Initialize Renderer
	renderer = new Renderer();
	renderer->allocateOnGPU(*scene);
	renderer->renderLoop();
	renderer->renderImage();
	renderer->free();

#ifdef ENABLE_VISUALIZER
	launch_visualizer(&(renderer->render_data));
#endif

}