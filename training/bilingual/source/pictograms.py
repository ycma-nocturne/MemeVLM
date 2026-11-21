"""Project-drawn pictograms for original synthetic meme scenes."""

from PIL import ImageDraw


def draw_icon(draw: ImageDraw.ImageDraw, name: str, x: int, y: int, color: str) -> None:
    """Draw a 100 × 100 logical icon with constant stroke and accent."""

    def line(*points, width=6):
        draw.line([(x + px, y + py) for px, py in points], fill=color, width=width, joint="curve")

    def oval(left, top, right, bottom, fill=False):
        draw.ellipse((x + left, y + top, x + right, y + bottom),
                     outline=color, fill=color if fill else None, width=5)

    def rect(left, top, right, bottom, fill=False):
        draw.rounded_rectangle((x + left, y + top, x + right, y + bottom),
                               radius=6, outline=color, fill=color if fill else None, width=5)

    def shape(*points, fill=False):
        coords = [(x + px, y + py) for px, py in points]
        if fill:
            draw.polygon(coords, fill=color)
        else:
            draw.line(coords + [coords[0]], fill=color, width=5, joint="curve")

    def dots(start_x, end_x, row_y):
        for px in range(start_x, end_x + 1, 11):
            oval(px, row_y, px + 3, row_y + 3, fill=True)

    if name in {"paper", "list", "patch", "break", "stamp", "book", "folder"}:
        if name == "book":
            shape((11, 22), (37, 19), (50, 27), (64, 19), (90, 23), (90, 77),
                  (65, 74), (50, 82), (36, 74), (11, 77))
            line((50, 29), (50, 78), width=5)
            for ox in (20, 59):
                line((ox, 42), (ox + 20, 43), width=4)
                line((ox, 56), (ox + 20, 57), width=4)
        elif name == "folder":
            shape((10, 76), (10, 25), (40, 25), (50, 35), (88, 35), (88, 76))
            line((12, 48), (87, 48), width=5)
        elif name == "patch":
            shape((20, 60), (60, 20), (80, 40), (40, 80))
            shape((34, 46), (46, 34), (66, 54), (54, 66))
        elif name == "break":
            line((30, 83), (21, 83), (21, 15), (76, 15), (76, 83), (53, 83), width=5)
            line((42, 16), (51, 32), (39, 46), (55, 58), (42, 82), width=5)
        else:
            rect(20, 15, 78, 85)
            if name == "stamp":
                oval(37, 29, 61, 52)
                rect(33, 53, 65, 65, fill=True)
            else:
                for row_y in (37, 51, 65):
                    if name == "list":
                        oval(29, row_y - 2, 35, row_y + 4, fill=True)
                    line((40 if name == "list" else 31, row_y), (66, row_y), width=4)
    elif name in {"screen", "server", "remote", "toggle", "battery", "crop", "chart"}:
        if name == "screen":
            rect(12, 18, 88, 72)
            line((50, 72), (50, 85))
            line((30, 85), (70, 85))
        elif name == "server":
            for row_y in (19, 43, 67):
                rect(15, row_y, 85, row_y + 14)
                oval(25, row_y + 5, 29, row_y + 9, fill=True)
        elif name == "remote":
            rect(33, 10, 67, 90)
            oval(43, 23, 57, 37)
            for row_y in (50, 63, 76):
                dots(43, 54, row_y)
        elif name == "toggle":
            draw.rounded_rectangle((x+12, y+32, x+88, y+67), radius=17, outline=color, width=5)
            oval(19, 40, 39, 60, fill=True)
        elif name == "battery":
            rect(13, 30, 80, 71)
            rect(80, 42, 88, 59, fill=True)
            rect(20, 37, 48, 64, fill=True)
        elif name == "crop":
            line((18, 30), (70, 30), (70, 82))
            line((31, 18), (31, 70), (83, 70))
        else:
            line((17, 19), (17, 82), (88, 82))
            line((30, 68), (47, 47), (62, 57), (84, 27), width=7)
    elif name in {"cloud", "sun", "snow", "drop", "wind", "water", "planet", "mountain", "rock"}:
        if name == "cloud":
            line((27, 70), (75, 70), width=7)
            oval(12, 43, 43, 71)
            oval(30, 26, 64, 60)
            oval(60, 38, 88, 70)
        elif name == "sun":
            oval(33, 33, 67, 67, fill=True)
            for points in (((50, 11), (50, 22)), ((50, 78), (50, 89)), ((11, 50), (22, 50)),
                           ((78, 50), (89, 50)), ((21, 21), (29, 29)), ((71, 71), (79, 79)),
                           ((71, 29), (79, 21)), ((21, 79), (29, 71))):
                line(*points)
        elif name == "snow":
            for points in (((50, 13), (50, 87)), ((19, 31), (81, 69)), ((19, 69), (81, 31))):
                line(*points)
            oval(42, 42, 58, 58, fill=True)
        elif name == "drop":
            shape((50, 12), (77, 52), (76, 70), (64, 84), (50, 89), (33, 84), (23, 70), (23, 52))
        elif name in {"wind", "water"}:
            for row_y in (36, 55, 74):
                line((11, row_y), (24, row_y-7), (37, row_y), (50, row_y+7),
                     (64, row_y), (78, row_y-7), (90, row_y), width=5)
        elif name == "planet":
            oval(24, 23, 76, 77, fill=True)
            line((9, 65), (37, 51), (64, 44), (90, 40), (75, 56), (59, 64), width=5)
            oval(75, 14, 83, 22, fill=True)
        elif name == "mountain":
            shape((10, 77), (37, 29), (46, 43), (61, 19), (91, 77))
        else:
            shape((10, 63), (22, 38), (52, 25), (72, 35), (91, 64), (73, 77), (30, 81))
    elif name in {"people", "person", "dog", "cat", "mouse", "hand", "hat", "shoe"}:
        if name in {"people", "person"}:
            centers = (29, 68) if name == "people" else (50,)
            for center in centers:
                oval(center-11, 20, center+11, 42, fill=True)
                rect(center-15, 49, center+15, 81)
        elif name in {"dog", "cat", "mouse"}:
            if name == "mouse":
                oval(15, 19, 42, 45, fill=True)
                oval(58, 19, 85, 45, fill=True)
            elif name == "cat":
                shape((20, 53), (24, 17), (44, 39), fill=True)
                shape((56, 39), (77, 17), (80, 53), fill=True)
            else:
                rect(15, 31, 32, 70, fill=True)
                rect(68, 31, 85, 70, fill=True)
            oval(28, 35, 72, 77)
            for px in (40, 56):
                oval(px, 48, px + 5, 53, fill=True)
            oval(47, 59, 54, 65, fill=True)
        elif name == "hand":
            shape((41, 83), (22, 61), (27, 54), (39, 61), (39, 23), (46, 17),
                  (52, 23), (52, 46), (60, 33), (66, 37), (73, 34), (80, 39), (79, 69), (70, 83))
        elif name == "hat":
            shape((22, 67), (27, 33), (44, 25), (66, 28), (77, 68))
            line((11, 71), (90, 71), width=7)
        else:
            shape((12, 69), (23, 55), (36, 58), (50, 45), (63, 29), (72, 34), (83, 59),
                  (90, 65), (87, 78), (13, 78))
    elif name in {"bag", "box", "gift", "bottle", "cup", "bowl", "bread", "fruit", "cookie", "salt", "fork"}:
        if name in {"bag", "box", "gift"}:
            rect(19, 38 if name == "bag" else 27, 81, 81)
            if name == "bag":
                draw.arc((x+34, y+13, x+66, y+57), 180, 360, fill=color, width=5)
            else:
                line((19, 47), (81, 47), width=5)
                if name == "gift":
                    line((50, 27), (50, 81), width=5)
                    oval(27, 14, 50, 29)
                    oval(50, 13, 73, 28)
        elif name == "bottle":
            shape((42, 13), (58, 13), (58, 30), (67, 37), (67, 83), (33, 83), (33, 37), (42, 30))
            line((34, 58), (66, 58), width=5)
        elif name == "cup":
            shape((18, 36), (70, 36), (66, 73), (28, 73))
            draw.arc((x+56, y+37, x+91, y+68), 258, 100, fill=color, width=5)
            line((15, 83), (77, 83), width=5)
        elif name == "bowl":
            line((13, 42), (87, 42), width=5)
            draw.arc((x+13, y+15, x+87, y+82), 0, 180, fill=color, width=5)
        elif name in {"bread", "cookie", "fruit"}:
            if name == "bread":
                rect(12, 26, 87, 74)
                line((29, 42), (36, 55), width=4)
                line((49, 39), (56, 53), width=4)
                line((68, 42), (75, 55), width=4)
            elif name == "cookie":
                oval(18, 18, 82, 82)
                dots(31, 64, 36)
                dots(38, 61, 52)
                dots(32, 55, 63)
            else:
                oval(20, 30, 80, 83)
                line((50, 34), (50, 19), (65, 14), width=5)
        elif name == "salt":
            rect(31, 30, 69, 82)
            dots(39, 61, 37)
            line((33, 52), (67, 52), width=5)
        else:
            for px in (31, 43, 55):
                line((px, 17), (px, 45), width=5)
            line((31, 45), (55, 45), (43, 55), (43, 84))
            line((72, 17), (72, 84), width=7)
    elif name in {"car", "train", "bike", "wheel", "boat", "plane", "route"}:
        if name == "car":
            shape((15, 64), (19, 45), (31, 44), (39, 29), (67, 29), (77, 45), (87, 48), (89, 65))
            for px in (24, 69):
                oval(px, 61, px+12, 74, fill=True)
        elif name == "train":
            rect(25, 14, 75, 72)
            rect(32, 25, 68, 45)
            for px in (34, 60):
                oval(px, 54, px+6, 60, fill=True)
            line((38, 74), (26, 87))
            line((62, 74), (74, 87))
        elif name == "bike":
            oval(11, 51, 36, 76)
            oval(66, 51, 91, 76)
            shape((24, 63), (41, 42), (61, 61), (38, 63))
            line((62, 62), (75, 41), (80, 63), width=5)
        elif name == "wheel":
            oval(16, 16, 84, 84)
            for points in (((50, 19), (50, 81)), ((19, 50), (81, 50)), ((28, 28), (72, 72))):
                line(*points, width=4)
        elif name == "boat":
            shape((15, 66), (85, 66), (70, 82), (30, 82))
            shape((47, 21), (47, 59), (22, 59))
        elif name == "plane":
            shape((49, 11), (56, 19), (57, 43), (88, 60), (88, 69), (57, 62), (56, 79),
                  (65, 86), (65, 92), (49, 87), (34, 92), (34, 87), (43, 79), (43, 62),
                  (13, 69), (13, 60), (43, 43), (43, 19), fill=True)
        else:
            line((18, 76), (33, 53), (57, 64), (78, 32))
            oval(12, 71, 23, 82, fill=True)
            oval(72, 26, 84, 38, fill=True)
    elif name in {"clock", "music", "mail", "money", "key", "signal", "plug", "bolt", "bulb"}:
        if name == "clock":
            oval(16, 16, 84, 84)
            line((50, 27), (50, 51), (67, 60))
        elif name == "music":
            line((41, 71), (41, 28), (73, 21), (73, 62))
            oval(22, 64, 42, 80, fill=True)
            oval(54, 55, 74, 71, fill=True)
        elif name == "mail":
            rect(13, 25, 87, 75)
            line((17, 30), (50, 55), (83, 30), width=5)
        elif name == "money":
            rect(11, 26, 89, 74)
            oval(38, 39, 62, 62)
            dots(20, 20, 46)
            dots(75, 75, 46)
        elif name == "key":
            oval(16, 28, 47, 59)
            line((43, 54), (80, 80), (80, 69))
            line((69, 72), (70, 61))
        elif name == "signal":
            for offset in (0, 13, 27):
                draw.arc((x+9+offset, y+21+offset, x+91-offset, y+94-offset), 218, 322,
                         fill=color, width=5)
            oval(45, 65, 55, 75, fill=True)
        elif name == "plug":
            rect(28, 35, 72, 65)
            line((42, 36), (42, 17))
            line((58, 36), (58, 17))
            line((50, 65), (50, 87))
        elif name == "bulb":
            oval(29, 16, 71, 59)
            line((36, 54), (40, 73), (60, 73), (64, 54))
            line((43, 84), (57, 84))
        else:
            shape((53, 11), (22, 56), (47, 56), (43, 89), (78, 42), (53, 42), fill=True)
    elif name in {"down", "up", "refresh", "swap"}:
        if name in {"up", "down"}:
            start, end = (77, 25) if name == "up" else (25, 77)
            line((50, start), (50, end), width=8)
            head = end + (19 if name == "up" else -19)
            line((30, head), (50, end), (70, head), width=8)
        elif name == "refresh":
            draw.arc((x+18, y+18, x+82, y+82), -15, 300, fill=color, width=6)
            shape((73, 20), (75, 45), (53, 33), fill=True)
        else:
            line((16, 36), (82, 36), (67, 21), width=6)
            line((83, 64), (17, 64), (32, 80), width=6)
    elif name in {"plant", "home", "chair", "trophy", "umbrella", "heart", "star", "crown"}:
        if name == "plant":
            rect(31, 63, 70, 86)
            line((50, 65), (50, 31))
            oval(23, 30, 48, 50, fill=True)
            oval(52, 20, 77, 42, fill=True)
        elif name == "home":
            shape((14, 44), (50, 16), (86, 44), (77, 44), (77, 81), (23, 81), (23, 44))
            line((43, 81), (43, 61), (57, 61), (57, 81), width=5)
        elif name == "chair":
            line((30, 19), (30, 63), (71, 63), (71, 84))
            line((30, 62), (30, 84))
            line((31, 53), (70, 53), width=6)
        elif name == "trophy":
            shape((27, 20), (73, 20), (69, 51), (57, 62), (43, 62), (31, 51))
            line((50, 62), (50, 80))
            line((34, 82), (66, 82))
        elif name == "umbrella":
            draw.arc((x+15, y+21, x+85, y+79), 180, 360, fill=color, width=5)
            line((15, 50), (85, 50))
            line((50, 50), (50, 76), (61, 84), (67, 76))
        elif name == "heart":
            oval(19, 28, 56, 61, fill=True)
            oval(44, 28, 81, 61, fill=True)
            shape((21, 46), (79, 46), (50, 82), fill=True)
        elif name == "star":
            shape((50, 14), (60, 38), (87, 40), (67, 57), (73, 84), (50, 70), (26, 84),
                  (33, 57), (13, 40), (40, 38), fill=True)
        else:
            shape((21, 35), (42, 54), (50, 28), (61, 54), (80, 34), (72, 75), (28, 75), fill=True)
    else:
        raise ValueError(f"unknown pictogram: {name}")
