use vulkano::format::{Format, ClearValue};
use vulkano::image::{Dimensions, StorageImage};
use vulkano::instance::{Instance, InstanceExtensions, PhysicalDevice};
use vulkano::device::{Device, DeviceExtensions, Features};
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBuffer};
use vulkano::sync::GpuFuture;
use std::sync::Arc;
use vulkano::pipeline::ComputePipeline;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::descriptor::PipelineLayoutAbstract;
use image::{ImageBuffer, Rgba};
use vulkano::framebuffer::Framebuffer;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::framebuffer::Subpass;
use vulkano::command_buffer::DynamicState;
use vulkano::pipeline::viewport::Viewport;

pub fn graphics() {
  let instance = Instance::new(None, &InstanceExtensions::none(), None)
    .expect("failed to create instance");
  let physical = PhysicalDevice::enumerate(&instance).next().expect("no device available");

  println!("yes");
  println!("{:?}", instance);
  println!("{:?}", physical);

  for family in physical.queue_families() {
    println!("Family {:?} queue count: {:?}", family.id() , family.queues_count());
  }

  let queue_family = physical.queue_families()
    .find(|&q| q.supports_graphics())
    .expect("couldn't find a graphical queue family");

  let (device, mut queues) = {
    Device::new(physical, &Features::none(), &DeviceExtensions::supported_by_device(physical),
                [(queue_family, 0.5)].iter().cloned())
                .expect("failed to create device")
  };
  let queue = queues.next().unwrap();

  mod cs {
    vulkano_shaders::shader!{
    ty: "compute",
    src: "
          #version 450

          layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

          layout(set = 0, binding = 0, rgba8) uniform writeonly image2D img;

          void main() {
              vec2 norm_coordinates = (gl_GlobalInvocationID.xy + vec2(0.5)) / vec2(imageSize(img));
              vec2 c = (norm_coordinates - vec2(0.5)) * 2.0 - vec2(1.0, 0.0);

              vec2 z = vec2(0.0, 0.0);
              float i;
              for (i = 0.0; i < 1.0; i += 0.005) {
                  z = vec2(
                      z.x * z.x - z.y * z.y + c.x,
                      z.y * z.x + z.x * z.y + c.y
                  );

                  if (length(z) > 4.0) {
                      break;
                  }
              }

              vec4 to_write = vec4(vec3(i), 1.0);
              imageStore(img, ivec2(gl_GlobalInvocationID.xy), to_write);
          }
      "
    }
  }

  let image = StorageImage::new(device.clone(), Dimensions::Dim2d { width: 1024, height: 1024 }, Format::R8G8B8A8Unorm, Some(queue.family())).unwrap();

  let shader = cs::Shader::load(device.clone())
        .expect("failed to create shader module");

  let compute_pipeline = Arc::new(ComputePipeline::new(device.clone(), &shader.main_entry_point(),
                                                        &()).expect("failed to create compute pipeline"));

  let layout = compute_pipeline.layout().descriptor_set_layout(0).unwrap();
  let set = Arc::new(PersistentDescriptorSet::start(layout.clone())
    .add_image(image.clone()).unwrap()
    .build().unwrap()
  );

  let buf = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false,
                                        (0 .. 1024 * 1024 * 4).map(|_| 0u8))
                                        .expect("failed to create buffer");


  let mut builder = AutoCommandBufferBuilder::new(device.clone(), queue.family()).unwrap();
  builder
      .dispatch([1024 / 8, 1024 / 8, 1], compute_pipeline.clone(), set.clone(), ()).unwrap()
      .copy_image_to_buffer(image.clone(), buf.clone()).unwrap();
  let command_buffer = builder.build().unwrap();


  let finished = command_buffer.execute(queue.clone()).unwrap();
  finished.then_signal_fence_and_flush().unwrap()
    .wait(None).unwrap();

  //let buffer_content = buf.read().unwrap();
  //let image = ImageBuffer::<Rgba<u8>, _>::from_raw(1024, 1024, &buffer_content[..]).unwrap();
  //image.save("image.png").unwrap();
  #[derive(Default, Copy, Clone)]
  struct Vertex {
    position: [f32; 2],
  }

  vulkano::impl_vertex!(Vertex, position);
  let vertex1 = Vertex { position: [-0.5, -0.5] };
  let vertex2 = Vertex { position: [ 0.0,  0.5] };
  let vertex3 = Vertex { position: [ 0.5, -0.25] };

  let vertex_buffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, 
                                                                          vec![vertex1, vertex2, vertex3].into_iter()).unwrap();
  mod vs {
    vulkano_shaders::shader!{
      ty: "vertex",
      src: "
          #version 450
  
          layout(location = 0) in vec2 position;
          
          void main() {
              gl_Position = vec4(position, 0.0, 1.0);
          }
      "
    }
  }

  mod fs {
    vulkano_shaders::shader!{
      ty: "fragment",
      src: "
        #version 450

        layout(location = 0) out vec4 f_color;
        
        void main() {
            f_color = vec4(1.0, 0.0, 0.0, 1.0);
        }      
      "
    }
  }

  let vs = vs::Shader::load(device.clone()).expect("failed to create shader module");
  let fs = fs::Shader::load(device.clone()).expect("failed to create shader module");

  let render_pass = Arc::new(vulkano::single_pass_renderpass!(device.clone(),
    attachments: {
        color: {
            load: Clear,
            store: Store,
            format: Format::R8G8B8A8Unorm,
            samples: 1,
        }
    },
    pass: {
        color: [color],
        depth_stencil: {}
    }
  ).unwrap());


  let framebuffer = Arc::new(Framebuffer::start(render_pass.clone())
      .add(image.clone()).unwrap()
      .build().unwrap());

  let mut builder = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family()).unwrap();
  builder
      .begin_render_pass(framebuffer.clone(), false, vec![[0.0, 0.0, 1.0, 1.0].into()])
      .unwrap()
      .end_render_pass()
      .unwrap();

  
  let pipeline = Arc::new(GraphicsPipeline::start()
  // Defines what kind of vertex input is expected.
  .vertex_input_single_buffer::<Vertex>()
  // The vertex shader.
  .vertex_shader(vs.main_entry_point(), ())
  // Defines the viewport (explanations below).
  .viewports_dynamic_scissors_irrelevant(1)
  // The fragment shader.
  .fragment_shader(fs.main_entry_point(), ())
  // This graphics pipeline object concerns the first pass of the render pass.
  .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
  // Now that everything is specified, we call `build`.
  .build(device.clone())
  .unwrap());

  let dynamic_state = DynamicState {
    viewports: Some(vec![Viewport {
        origin: [0.0, 0.0],
        dimensions: [1024.0, 1024.0],
        depth_range: 0.0 .. 1.0,
    }]),
    .. DynamicState::none()
  };

  let mut builder = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family()).unwrap();

  builder
    .begin_render_pass(framebuffer.clone(), false, vec![[0.0, 0.0, 1.0, 1.0].into()])
    .unwrap()

    .draw(pipeline.clone(), &dynamic_state, vertex_buffer.clone(), (), ())
    .unwrap()

    .end_render_pass()
    .unwrap()

    .copy_image_to_buffer(image.clone(), buf.clone())
    .unwrap();

  let command_buffer = builder.build().unwrap();

  let finished = command_buffer.execute(queue.clone()).unwrap();
  finished.then_signal_fence_and_flush().unwrap()
      .wait(None).unwrap();

  let buffer_content = buf.read().unwrap();
  let image = ImageBuffer::<Rgba<u8>, _>::from_raw(1024, 1024, &buffer_content[..]).unwrap();
  image.save("triangle.png").unwrap();
}