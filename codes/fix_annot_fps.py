import xml.etree.ElementTree as ET
from math import ceil
"""
This code was used to fix annotations when we changed the videos fps to 24fps.
"""

video_name = 'deleteme'

filename = f'/home/wesley.passos/repos/mosquitoes-wes/data/v1.0/annotation/{video_name}.xml'

tree = ET.parse(filename)
root = tree.getroot()

stop_frame = int(tree.find('.//stop_frame').text)
tree.find('.//stop_frame').text = str(round(stop_frame / 2))

tracks = root.findall('track')

# iterate over tracks
for track in tracks:
    boxes = track.findall('box')

    # iterate over bboxes
    for box in boxes:
        info = box.attrib
        frame = int(info['frame'])

        if frame % 2 == 0 and frame != 0:
            track.remove(box)

        else:
            if frame == 1:
                continue
            box.set("frame", f"{int(ceil(frame/2))}")

    # set the last one as 1
    boxes = track.findall('box')
    boxes[-1].set("outside", "1")

# tree.write(f'{video_name}_fps.xml',
#            encoding="utf-8",
#            xml_declaration=True,
#            short_empty_elements=False)
