# Example taken from Movidius Neural Compute Stick
# See AlexNet Example
# The model graph must be generated using the command mvNCProfile

from mvnc import mvncapi as mvnc


def main():
    print('Initializing Main function')

    # Loading memory stick instances
    devices = mvnc.EnumerateDevices()
    if len(devices) == 0:
        raise Exception('No devices found')

    print('Total devices found: {0}'.format(len(devices)))

    # Select first device
    device = devices[0]

    net_blob_path = '/home/mauricio/Programs/openpose/openpose/models/pose/body_25/graph'
    with open(net_blob_path, mode='rb') as f:
        blob = f.read()

    graph = device.AllocateGraph(blob)

    # Cleaning up the graph and close the device
    graph.DeallocateGraph()
    device.CloseDevice()




if __name__ == '__main__':
    main()
