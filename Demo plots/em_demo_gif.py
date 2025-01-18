import imageio
images = []
filenames = ['learned_cats_example_init.png','learned_cats_example_init.png','learned_cats_example_init.png',
             'learned_cats_example_3.png','learned_cats_example_3.png','learned_cats_example_3.png',
             'learned_cats_example_10.png', 'learned_cats_example_10.png', 'learned_cats_example_10.png',
             'learned_cats_example_15.png','learned_cats_example_15.png','learned_cats_example_15.png',
'learned_cats_example_20.png','learned_cats_example_20.png','learned_cats_example_20.png',
'learned_cats_example_30.png','learned_cats_example_30.png','learned_cats_example_30.png',
'learned_cats_example_40.png','learned_cats_example_40.png','learned_cats_example_40.png',
'learned_cats_example_50.png','learned_cats_example_50.png','learned_cats_example_50.png',
'learned_cats_example_100.png','learned_cats_example_100.png','learned_cats_example_100.png',
             ]
for filename in filenames:
    images.append(imageio.imread(filename))
kargs= {'duration':250}
imageio.mimsave('em_search_example.gif', images,'GIF', loop=0,**kargs)