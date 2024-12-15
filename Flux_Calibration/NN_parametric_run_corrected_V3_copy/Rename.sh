for file in *; do
  new_name=$(echo "$file" | sed -E 's/Iter:([0-9]+)\.0/Iter:\1/')
  mv "$file" "$new_name"
done

