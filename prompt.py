TEMPLATES = ['a cropped photo of the {}.', 'a cropped photo of a {}.', 'a close-up photo of a {}.', 'a close-up photo of the {}.', \
    'a bright photo of a {}.', 'a bright photo of the {}.', 'a dark photo of the {}.', 'a dark photo of a {}.',\
        'a jpeg corrupted photo of a {}.', 'a jpeg corrupted photo of the {}.', 'a blurry photo of the {}.', 'a blurry photo of a {}.',\
            'a photo of the {}', 'a photo of a {}', 'a photo of a small {}', 'a photo of the small {}', 'a photo of a large {}',\
                'a photo of the large {}', 'a photo of the {} for visual inspection.', 'a photo of a {} for visual inspection.',\
                    'a photo of the {} for anomaly detection.', 'a photo of a {} for anomaly detection.']


REAL_NAME = {'Brain': 'Brain', 'Liver':'Liver', 'Retina_RESC':'retinal OCT', 'Chest':'Chest X-ray film', 'Retina_OCT2017':'retinal OCT', 'Histopathology':'histopathological image'}

# Simple organ aliases for user-friendly input.
ORGAN_ALIASES = {
    'brain': 'Brain',
    'liver': 'Liver',
    'retina': 'Retina_OCT2017',
    'retina_resc': 'Retina_RESC',
    'retina_oct2017': 'Retina_OCT2017',
    'chest': 'Chest',
    'histopathology': 'Histopathology',
}

