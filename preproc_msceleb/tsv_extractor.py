import os
import base64
import struct

#fid = open("/home/prithvi/dsets/MSCeleb/FaceImageCroppedWithAlignment.tsv", "r")
#base_path = '/home/prithvi/dsets/MSCeleb/raw'
#n_ppl = 10

def extract(tsv_path,base_path,n_ppl):
    fid = open(tsv_path, 'r')
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    bbox_file = open(base_path + '/bboxes.txt', 'w')
    i = 0
    while i < n_ppl:
        line = fid.readline()
        if line:
            data_info = line.split('\t')
            filename = data_info[0] + "/" + data_info[1] + "_" + data_info[4] + ".jpg"
            bbox = struct.unpack('ffff', data_info[5].decode("base64"))
            bbox_file.write(filename + " "+ (" ".join(str(bbox_value) for bbox_value  in bbox)) + "\n")
            img_data = data_info[6].decode("base64")
            output_file_path = base_path + "/" + filename 
            if os.path.exists(output_file_path):
                print output_file_path + " exists"
            output_path = os.path.dirname(output_file_path)
            if not os.path.exists(output_path):
                os.mkdir(output_path)
                i += 1
            img_file = open(output_file_path, 'w')
            img_file.write(img_data)
            img_file.close()
        else:
            break
    bbox_file.close()
    fid.close()
