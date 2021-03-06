import xml.etree.cElementTree as ElementTree
import numpy as np
import os
import pickle as pkl


class XmlListConfig(list):
    def __init__(self, aList):
        for element in aList:
            if element:
                # treat like dict
                if len(element) == 1 or element[0].tag != element[1].tag:
                    self.append(XmlDictConfig(element))
                # treat like list
                elif element[0].tag == element[1].tag:
                    self.append(XmlListConfig(element))
            elif element.text:
                text = element.text.strip()
                if text:
                    self.append(text)


class XmlDictConfig(dict):
    '''
    Example usage:

    >>> tree = ElementTree.parse('your_file.xml')
    >>> root = tree.getroot()
    >>> xmldict = XmlDictConfig(root)

    Or, if you want to use an XML string:

    >>> root = ElementTree.XML(xml_string)
    >>> xmldict = XmlDictConfig(root)

    And then use xmldict for what it is... a dict.
    '''

    def __init__(self, parent_element):
        childrenNames = []
        for child in parent_element.getchildren():
            childrenNames.append(child.tag)

        if parent_element.items():  # attributes
            self.update(dict(parent_element.items()))
        for element in parent_element:
            if element:
                # treat like dict - we assume that if the first two tags
                # in a series are different, then they are all different.
                # print len(element), element[0].tag, element[1].tag
                if len(element) == 1 or element[0].tag != element[1].tag:
                    aDict = XmlDictConfig(element)
                # treat like list - we assume that if the first two tags
                # in a series are the same, then the rest are the same.
                else:
                    # here, we put the list in dictionary; the key is the
                    # tag name the list elements all share in common, and
                    # the value is the list itself
                    aDict = {element[0].tag: XmlListConfig(element)}
                # if the tag has attributes, add those to the dict
                if element.items():
                    aDict.update(dict(element.items()))

                if childrenNames.count(element.tag) > 1:
                    try:
                        currentValue = self[element.tag]
                        currentValue.append(aDict)
                        self.update({element.tag: currentValue})
                    except:  # the first of its kind, an empty list must be created
                        self.update({element.tag: [aDict]})  # aDict is written in [], i.e. it will be a list

                else:
                    self.update({element.tag: aDict})
            # this assumes that if you've got an attribute in a tag,
            # you won't be having any text. This may or may not be a
            # good idea -- time will tell. It works for the way we are
            # currently doing XML configuration files...
            elif element.items():
                self.update({element.tag: dict(element.items())})
            # finally, if there are no child tags and no attributes, extract
            # the text
            else:
                self.update({element.tag: element.text})


# Class definition ends
# ---------------------------------------------------------------------------------------------------------------------

## xmltopickle Converter. pickle format: xmin,ymax,xmax,ymin,c

def main(directory_name):
    path = os.getcwd() + directory_name
    file_count = 0
    for file in os.listdir(path):
        extension = file.split('.')[-1]
        filename = str(file.split('.')[0])
        new_file_loc = path + filename
        if extension == 'xml':
            file_count += 1
            file_loc = path + file
            tree = ElementTree.parse(file_loc)
            root = tree.getroot()
            xmldict = XmlDictConfig(root)
            boxes = np.empty((0, 5))
            for keys in xmldict:
                if keys == "object":
                    if str(type(xmldict[keys])) == "<class 'data_aug_helper.XmlListConfig.XmlDictConfig'>":
                        xmin = float(xmldict[keys]["bndbox"]["xmin"])
                        ymax = float(xmldict[keys]["bndbox"]["ymax"])
                        xmax = float(xmldict[keys]["bndbox"]["xmax"])
                        ymin = float(xmldict[keys]["bndbox"]["ymin"])
                        classname = xmldict[keys]["name"]
                        class_id = class_text_to_int(classname)
                        box = []
                        box.append([xmin, ymax, xmax, ymin, class_id])
                        boxes = np.append(boxes, box, axis=0)
                    else:
                        for object in xmldict[keys]:
                            xmin = float(object["bndbox"]["xmin"])
                            ymax = float(object["bndbox"]["ymax"])
                            xmax = float(object["bndbox"]["xmax"])
                            ymin = float(object["bndbox"]["ymin"])
                            classname = object["name"]
                            class_id = class_text_to_int(classname)
                            box = []
                            box.append([xmin, ymax, xmax, ymin, class_id])
                            boxes = np.append(boxes, box, axis=0)
            pkl.dump(boxes, open(new_file_loc + ".pkl", "wb"))
    print("\nFinished...\nFile Count:", file_count)


def class_text_to_int(row_label):
    if row_label.__eq__('sugar_beet'):
        return 1
    else:
        None


if __name__ == "__main__":
    dir_name = '/imgs/train/'
    main(dir_name)
