import html
from html.parser import HTMLParser
from html.entities import name2codepoint
import os
import sys

class VueJSComponentSplitter(HTMLParser):
    def __init__(self, filename, *args, **kwargs):
        super(VueJSComponentSplitter, self).__init__(*args, **kwargs)
        self.stack_data = [""]
        self.tag_depth = 0
        self.tag_stack = []
        self.filename = filename
        if not filename.endswith(".vue"):
            raise ValueError("Filename must have extension .vue")

    def handle_starttag(self, tag, attrs):
        self.tag_depth += 1
        attributes = " ".join("{0}='{1}'".format(attr[0], attr[1]) for attr in attrs)
        if len(attributes) > 0:
            attributes = " " + attributes
        tag_data = "<{0}{1}>".format(tag, attributes)
        if self.tag_depth == 1:
            self.stack_data.append("")
            self.tag_stack = [tag, {attr[0]: attr[1] for attr in attrs}]
        else:
            self.stack_data.append(tag_data)
        
    def process_tag(self, data):
        tagname = self.tag_stack[0]
        attributes = self.tag_stack[1]
        if tagname not in ["script", "style", "template"]:
            raise ValueError("Unrecognized tagname: %s" % tagname)
        basedir = self.filename.replace(".vue", "")
        if not os.path.exists(basedir): os.makedirs(basedir)
        name_dict = {"script": "script.js", "style": "style.css", "template": "template.html"}
        name = name_dict[tagname]
        open(basedir + "/" + name, "w").write(data)
    
    def handle_endtag(self, tag):
        self.tag_depth -= 1
        content = self.stack_data.pop()
        self.stack_data[-1] = "{0}{1}".format(self.stack_data[-1], content)
        if self.tag_depth == 0:
            self.tag_stack.append(tag)
            self.process_tag(self.stack_data[-1])
            self.stack_data.append("")
        else:
            self.stack_data[-1] = "{0}</{1}>".format(self.stack_data[-1], tag)

    def handle_data(self, data):
        data = html.escape(data)
        if len(self.stack_data) == 0:
            self.stack_data.append("") 
        self.stack_data[-1] = "{0}{1}".format(self.stack_data[-1], data)

    def handle_comment(self, data):
        self.stack_data[-1] = "{0}<!--{1}-->".format(self.stack_data[-1], data)

    @classmethod
    def parse_files(cls, filelist):
        for filename in filelist:
            parser = cls(filename=filename)
            parser.feed(open(filename, "r").read())
            parser.close()


current_dir = os.path.dirname(sys.argv[0])
VueJSComponentSplitter.parse_files([current_dir + "/" + filename
                                    for filename in os.listdir(current_dir) if filename.endswith(".vue")])

