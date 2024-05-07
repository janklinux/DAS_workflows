# -*- coding: utf-8 -*-

from .context import das_workflows

import unittest


class AdvancedTestSuite(unittest.TestCase):
    """Advanced test cases."""

    def test_thoughts(self):
        self.assertIsNone(das_workflows.hmm())


if __name__ == '__main__':
    unittest.main()
