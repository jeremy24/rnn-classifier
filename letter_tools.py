import glob
import imghdr
import os

import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw


def _is_ascii(s):
	return all(ord(c) < 128 for c in s)


def text2png(text, fullpath, color="#FFF", bgcolor="#000",
			 fontfullpath="fonts/Oswald-Bold.ttf",
			 fontsize=35, leftpadding=3, rightpadding=3,
			 width=20, height=None):
	REPLACEMENT_CHARACTER = u'\uFFFD'
	NEWLINE_REPLACEMENT_STRING = ' ' + REPLACEMENT_CHARACTER + ' '

	font = ImageFont.load_default() if fontfullpath == None else ImageFont.truetype(fontfullpath, fontsize)
	text = text.replace('\n', NEWLINE_REPLACEMENT_STRING)

	lines = []
	line = u""

	can_use = [".", ",", "/", "[", "]", "(", ")", ""]

	print("\nProvided text:", text)


	print("\tUsing: ", text)
	if len(text) == 0:
		print("\tNo valid text, bailing out...")
		return

	for word in text.split():
		print(word)
		if word == REPLACEMENT_CHARACTER:  # give a blank line
			lines.append(line[1:])  # slice the white space in the begining of the line
			line = u""
			lines.append(u"")  # the blank line
		elif font.getsize(line + ' ' + word)[0] <= (width - rightpadding - leftpadding):
			line += ' ' + word
		else:  # start a new line
			lines.append(line[1:])  # slice the white space in the begining of the line
			line = u""

			# TODO: handle too long words at this point
			line += ' ' + word  # for now, assume no word alone can exceed the line width

	if len(line) != 0:
		lines.append(line[1:])  # add the last line

	line_height = font.getsize(text)[1]

	width = font.getsize(text)[0]
	width += int( width * .10 )

	if height is not None:
		line_height = height

	img_height = line_height * (len(lines) + 1)


	img = Image.new("RGBA", (width, img_height), bgcolor)
	draw = ImageDraw.Draw(img)

	y = 0
	for line in lines:
		draw.text((leftpadding, y), line, color, font=font)
		y += line_height

	print("\tSaving to: ", fullpath)
	img.save(fullpath)
