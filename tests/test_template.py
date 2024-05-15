import template
import template.command_line


def test_example():
    """
    Test the example function

    """
    assert template.example(1, 2) == 3


def test_command_line_main():
    """
    Test the command line

    """
    template.command_line.main(["-a", "1", "-b", "2"])
