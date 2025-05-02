cp -r src tmp
find tmp -name "*.py" ! -name "lexicon.py" -exec sed -i "" "s/support/scales_nlp_support/g" {} \;

python setup.py bdist_wheel --universal
twine upload dist/*
rm -r tmp build dist