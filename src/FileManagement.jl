module FileManagement

using Base.Filesystem

export setupfolder
export numfiles


function setupfolder(folder_path::String)
    if !isdir(folder_path)
      mkdir(folder_path)
    else
      rm(folder_path,recursive=true)
      mkdir(folder_path)
    end
  end
  
  
  function numfiles(foldername::String)
    files_and_dirs = readdir(foldername)  # reading files and directory
    num::Int64 = 0
    for i in files_and_dirs
        fullpath = joinpath(foldername, i)  # join foldername with file/directory name
        if isfile(fullpath)
            num += 1
        end
    end
    return num
  end



end