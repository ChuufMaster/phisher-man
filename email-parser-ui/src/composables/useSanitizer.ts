import DOMPurify from 'dompurify'

export const sanitize = (input: string): string => DOMPurify.sanitize(input)
