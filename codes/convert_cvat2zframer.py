import xml.etree.ElementTree as ET

# TODO: HARDCODED -- argparse
filename = 'data/_under_construction/20190601_rectfied_DJI_0006/20190601_rectfied_DJI_0006.xml'
zfile = open('data/_under_construction/20190601_rectfied_DJI_0006/20190601_rectfied_DJI_0006.txt',
             "w")
zfile.write("ZMARKER100\n\n")

tree = ET.parse(filename)
root = tree.getroot()

objs = root.findall('track')
zfile.write("#  ==============================================================================\n")
zfile.write(f"NOBJECTS:{len(objs)}\n\n")

# TODO: HARDCODED -- get video resolution
zfile.write(f"FRAMESXY:4096x2160\n\n")

# iterate over objs
for obj in objs:
    zfile.write("BOBJECT\n\n")
    zfile.write("SHAPE:1\n\n")
    # TODO: COLOR -- change color per objects
    zfile.write(f"COLOR:{0}, {0}, {255}\n\n")

    obj_label = obj.attrib['label']
    obj_id = obj.attrib['id']

    # TODO: include attribute in object name
    zfile.write(f"NAME:{obj_label}-{obj_id}\n\n")

    # iterate over bbs
    for bb in obj:
        info = bb.attrib

        frame = info['frame']
        xtl = float(info['xtl'])
        xbr = float(info['xbr'])
        ytl = float(info['ytl'])
        ybr = float(info['ybr'])

        zfile.write(f"RECT:  {frame}, {int(xtl)}, {int(ytl)}, {int(xbr)}, {int(ybr)}\n")

        # # iterate over attributes
        # for att in bb:
        #     att_name = att.attrib['name']
        #     att_value = att.text
        #     print()

    zfile.write("\nEOBJECT\n\n")

zfile.write("END\n\n")
zfile.close()

print()
