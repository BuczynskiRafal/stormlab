"""Model string representations must not assume fields the custom User lacks."""

from django.test import TestCase

from .factories import make_swmm_model, make_user


class SwmmModelStrTests(TestCase):
    def test_str_uses_the_email_based_username(self):
        user = make_user()
        model = make_swmm_model(user)

        text = str(model)

        self.assertIn(user.get_username(), text)
        self.assertIn(model.file.name, text)
