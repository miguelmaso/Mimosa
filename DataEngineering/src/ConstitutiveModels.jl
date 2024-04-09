module ConstitutiveModels

export logreg

# Jacobian regularization
function logreg(J; Threshold=0.01)
    if J>= Threshold
      return log(J)
    else
      return log(Threshold)-(3.0/2.0)+(2/Threshold)*J-(1/(2*Threshold^2))*J^2
    end
  end
  


end