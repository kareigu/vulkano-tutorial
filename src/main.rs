use vulkano::instance::Instance;
use vulkano::instance::InstanceExtensions;
use vulkano::instance::PhysicalDevice;
use vulkano::device::Device;
use vulkano::device::DeviceExtensions;
use vulkano::device::Features;
use vulkano::buffer::BufferUsage;
use vulkano::buffer::CpuAccessibleBuffer;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::command_buffer::CommandBuffer;
use vulkano::sync::GpuFuture;
use std::sync::Arc;
use vulkano::pipeline::ComputePipeline;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::descriptor::PipelineLayoutAbstract;


fn main() {
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

    struct DataStruct {
        a: u32,
        b: bool,
    }

    let data = DataStruct{a: 12, b: true};
    let databuffer = CpuAccessibleBuffer::from_data(device.clone(), BufferUsage::all(),
                                                true, data).expect("failed to create a buffer");

    let iter = (0 .. 128).map(|_| 5u8);
    let iterbuffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(),
                                                    true, iter).expect("failed to create a buffer");
    let mut datacontent = databuffer.write().unwrap();
    datacontent.a += 2;
    datacontent.b = false;

    let mut itercontent = iterbuffer.write().unwrap();
    itercontent[23] = 2;
    itercontent[85] = 124;

    let source_content = 0 .. 64;
    let source = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), true,
                                                source_content).expect("failed to create buffer");

    let dest_content = (0 .. 64).map(|_| 0);
    let dest = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), true,
                                              dest_content).expect("failed to create buffer");

    let mut builder = AutoCommandBufferBuilder::new(device.clone(), queue.family()).unwrap();
    builder.copy_buffer(source.clone(), dest.clone()).unwrap();
    let command_buffer = builder.build().unwrap();

    let finished = command_buffer.execute(queue.clone()).unwrap();
    finished.then_signal_fence_and_flush().unwrap()
            .wait(None).unwrap();

    let src_content = source.read().unwrap();
    let dest_content = dest.read().unwrap();
    assert_eq!(&*src_content, &*dest_content);
    println!("{:?} \n {:?}", &*src_content, &*dest_content);


    let data_iter = 0 .. 65536;
    let data_buffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), true,
                                                 data_iter).expect("failed to create buffer");
    mod cs {
        vulkano_shaders::shader!{
        ty: "compute",
        src: "
            #version 450

            layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

            layout(set = 0, binding = 0) buffer Data {
                uint data[];
            } buf;

            void main() {
                uint idx = gl_GlobalInvocationID.x;
                buf.data[idx] *= 12;
            }"
        }
    }
    let shader = cs::Shader::load(device.clone())
        .expect("failed to create shader module");

    let compute_pipeline = Arc::new(ComputePipeline::new(device.clone(), &shader.main_entry_point(),
                                                        &()).expect("failed to create compute pipeline"));

    let set_layout = compute_pipeline.layout().descriptor_set_layout(0).unwrap();

    let set = Arc::new(PersistentDescriptorSet::start(set_layout.clone())
        .add_buffer(data_buffer.clone()).unwrap()
        .build().unwrap()
    );

    let mut builder = AutoCommandBufferBuilder::new(device.clone(), queue.family()).unwrap();
    builder.dispatch([1024, 1, 1], compute_pipeline.clone(), set.clone(), ()).unwrap();

    let command_buffer = builder.build().unwrap();

    let finished = command_buffer.execute(queue.clone()).unwrap();

    finished.then_signal_fence_and_flush().unwrap()
        .wait(None).unwrap();

    let content = data_buffer.read().unwrap();
    for (n, val) in content.iter().enumerate() {
        assert_eq!(*val, n as u32 * 12);
        //println!("N: {} - val: {}", n, *val);
    }
    
    println!("Everything succeeded!");
        
}
